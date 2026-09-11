"""
Repository layer: all database access flows through these classes.

Each repository owns its sessions: a session is opened, used, committed or
rolled back, and always closed within the same method. Endpoints never create
or leak sessions. Exceptions raised are domain-level (``PersistenceError`` /
``DuplicateRequestError``) so the API layer can map them to controlled
responses without exposing database internals.
"""

from __future__ import annotations

import datetime
from typing import Any, Optional, Sequence, Tuple

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.db.database import Database, utcnow_naive
from app.db.models import AuditLog, AuthToken, Screening, ScreeningFactor, User


class PersistenceError(RuntimeError):
    """Raised when a persistence operation cannot be completed."""


class DuplicateRequestError(PersistenceError):
    """Raised when a screening request_id already exists (unique constraint)."""


def _normalize_utc(value: object) -> datetime.datetime:
    """Coerce a user-supplied datetime to naive UTC for cross-dialect filtering."""
    if value is None:
        raise ValueError("Date filter cannot be empty")
    dt = value
    if not isinstance(dt, datetime.datetime):
        dt = datetime.datetime.fromisoformat(str(dt))
    if dt.tzinfo is not None:
        dt = dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return dt


def _severity(weight: int) -> str:
    if weight >= 30:
        return "HIGH"
    if weight >= 15:
        return "MEDIUM"
    return "LOW"


class ScreeningRepository:
    """CRUD, filtering, statistics, and reporting for screenings."""

    def __init__(self, database: Database) -> None:
        self._database = database

    # -- creation ---------------------------------------------------------
    def create(
        self,
        *,
        request_id: str,
        processing_time_ms: int,
        document_type: str,
        mrz_status: str,
        face_status: str,
        face_similarity: Optional[float],
        tampering_status: str,
        tampering_score: Optional[float],
        liveness_status: str = "NOT_CHECKED",
        liveness_score: Optional[float] = None,
        risk_score: int,
        risk_level: str,
        decision: str,
        status_color: str,
        module_states: dict[str, str],
        factor_list: Sequence[dict[str, Any]],
        mrz_source: str,
        user_id: Optional[int] = None,
        notes: Optional[str] = None,
        created_at: Optional[datetime.datetime] = None,
        audit_message: str = "",
    ) -> Screening:
        created_at = created_at or utcnow_naive()
        session = self._database.session()
        try:
            existing = session.execute(
                select(Screening.id).where(Screening.request_id == request_id)
            ).scalar_one_or_none()
            if existing is not None:
                raise DuplicateRequestError(f"request_id {request_id} already exists")

            screening = Screening(
                request_id=request_id,
                created_at=created_at,
                processing_time_ms=processing_time_ms,
                document_type=document_type,
                mrz_status=mrz_status,
                face_status=face_status,
                face_similarity=face_similarity,
                tampering_status=tampering_status,
                tampering_score=tampering_score,
                liveness_status=liveness_status,
                liveness_score=liveness_score,
                risk_score=risk_score,
                risk_level=risk_level,
                decision=decision,
                status_color=status_color,
                module_states=module_states,
                factors=[dict(f) for f in factor_list],
                mrz_source=mrz_source,
                user_id=user_id,
                notes=notes,
            )
            for factor in factor_list:
                name = str(factor.get("factor", "UNKNOWN"))
                weight = int(factor.get("weight") or 0)
                screening.factor_rows.append(ScreeningFactor(
                    factor_name=name,
                    severity=_severity(weight),
                    weight=weight,
                    description=str(factor.get("detail", "")),
                ))
            session.add(screening)
            # Flush so ScreeningFactor/AuditLog can reference screening.id.
            session.flush()
            session.add(AuditLog(
                screening_id=screening.id,
                event_type="screening.completed",
                created_at=created_at,
                request_id=request_id,
                message=audit_message or f"decision={decision}",
            ))
            session.commit()
            session.refresh(screening)
            return screening
        except DuplicateRequestError:
            session.rollback()
            raise
        except IntegrityError as exc:
            session.rollback()
            raise DuplicateRequestError(f"request_id {request_id} already exists") from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise PersistenceError(f"Failed to persist screening {request_id}") from exc
        finally:
            session.close()

    # -- reads ------------------------------------------------------------
    def get(self, screening_id: int) -> Optional[Screening]:
        with self._database.session() as session:
            return session.get(Screening, screening_id)

    def get_by_request_id(self, request_id: str) -> Optional[Screening]:
        with self._database.session() as session:
            return session.execute(
                select(Screening).where(Screening.request_id == request_id)
            ).scalar_one_or_none()

    def list(
        self,
        *,
        decision: Optional[str] = None,
        risk_level: Optional[str] = None,
        date_from: Optional[object] = None,
        date_to: Optional[object] = None,
        user_id: Optional[int] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Tuple[int, list[Screening]]:
        filters = []
        if decision:
            filters.append(Screening.decision == decision)
        if risk_level:
            filters.append(Screening.risk_level == risk_level)
        if date_from:
            filters.append(Screening.created_at >= _normalize_utc(date_from))
        if date_to:
            filters.append(Screening.created_at <= _normalize_utc(date_to))
        if user_id is not None:
            filters.append(Screening.user_id == user_id)

        with self._database.session() as session:
            total = session.execute(
                select(func.count()).select_from(Screening).where(*filters)
            ).scalar_one()
            rows = session.execute(
                select(Screening)
                .where(*filters)
                .order_by(Screening.id.desc())
                .limit(limit)
                .offset(offset)
            ).scalars().all()
            return int(total), list(rows)

    def list_factors(self, screening_id: int) -> list[ScreeningFactor]:
        with self._database.session() as session:
            rows = session.execute(
                select(ScreeningFactor)
                .where(ScreeningFactor.screening_id == screening_id)
                .order_by(ScreeningFactor.id)
            ).scalars().all()
            return list(rows)

    # -- aggregates -------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        with self._database.session() as session:
            total = self._count(session, None)
            cleared = self._count(session, Screening.decision == "CLEARED")
            high_risk = self._count(session, Screening.decision == "HIGH_RISK_REVIEW_REQUIRED")
            secondary = total - cleared - high_risk
            mrz_failures = self._count(session, Screening.mrz_status.in_(
                ["INVALID", "MALFORMED", "NOT_DETECTED", "OCR_FAILED", "OCR_LOW_CONFIDENCE"]))
            face_mismatches = self._count(session, Screening.face_status == "MISMATCH")
            suspicious = self._count(session, Screening.tampering_status == "SUSPICIOUS")
            by_decision = dict(
                session.execute(
                    select(Screening.decision, func.count())
                    .group_by(Screening.decision)
                ).all()
            )
            by_risk_level = dict(
                session.execute(
                    select(Screening.risk_level, func.count())
                    .group_by(Screening.risk_level)
                ).all()
            )
            return {
                "total": total,
                "cleared": cleared,
                "secondary_inspection": secondary,
                "high_risk": high_risk,
                "mrz_failures": mrz_failures,
                "face_mismatches": face_mismatches,
                "suspicious_tampering": suspicious,
                "by_decision": by_decision,
                "by_risk_level": by_risk_level,
            }

    def summary(self) -> dict[str, Any]:
        """Roll-up used by the legacy ``/api/v1/report/summary`` endpoint."""
        with self._database.session() as session:
            total = self._count(session, None)
            avg = session.execute(
                select(func.avg(Screening.processing_time_ms))
            ).scalar_one_or_none()
            cleared = self._count(session, Screening.decision == "CLEARED")
            high_risk = self._count(session, Screening.decision == "HIGH_RISK_REVIEW_REQUIRED")
            secondary = total - cleared - high_risk
            by_decision = dict(
                session.execute(
                    select(Screening.decision, func.count())
                    .group_by(Screening.decision)
                ).all()
            )
            by_risk_level = dict(
                session.execute(
                    select(Screening.risk_level, func.count())
                    .group_by(Screening.risk_level)
                ).all()
            )
            return {
                "total": total,
                "cleared": cleared,
                "secondary_inspection": secondary,
                "high_risk": high_risk,
                "avg_processing_time_ms": round(float(avg or 0.0), 1),
                "by_decision": by_decision,
                "by_risk_level": by_risk_level,
            }

    def update_decision(
        self,
        screening_id: int,
        decision: str,
        status_color: str,
        notes: Optional[str] = None,
        officer_id: Optional[int] = None,
    ) -> Optional[Screening]:
        with self._database.session() as session:
            screening = session.get(Screening, screening_id)
            if screening is None:
                return None
            screening.decision = decision
            screening.status_color = status_color
            if notes is not None:
                screening.notes = notes
            audit_msg = f"decision={decision}"
            if officer_id:
                audit_msg += f" officer_id={officer_id}"
            if notes:
                audit_msg += f" notes={notes}"
            session.add(AuditLog(
                screening_id=screening.id,
                event_type="decision.updated",
                created_at=utcnow_naive(),
                request_id=screening.request_id,
                message=audit_msg,
            ))
            session.commit()
            session.refresh(screening)
            return screening

    def trend(self, range_key: str = "24h", now: Optional[datetime.datetime] = None) -> dict[str, Any]:
        now = now or utcnow_naive()
        today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday_midnight = today_midnight - datetime.timedelta(days=1)

        if range_key == "7d":
            duration = datetime.timedelta(days=7)
            bucket_delta = datetime.timedelta(days=1)
            num_buckets = 7
            fmt = "%b %d"
        elif range_key == "30d":
            duration = datetime.timedelta(days=30)
            bucket_delta = datetime.timedelta(days=3)
            num_buckets = 10
            fmt = "%b %d"
        else:
            range_key = "24h"
            duration = datetime.timedelta(hours=24)
            bucket_delta = datetime.timedelta(hours=4)
            num_buckets = 6
            fmt = "%H:%M"

        range_start = now - duration
        prior_range_start = range_start - duration

        with self._database.session() as session:
            today_total = self._count(session, Screening.created_at >= today_midnight)
            yesterday_total = self._count(
                session,
                (Screening.created_at >= yesterday_midnight) & (Screening.created_at < today_midnight),
            )
            if yesterday_total > 0:
                delta_today_pct = round(((today_total - yesterday_total) / yesterday_total) * 100.0, 1)
            elif today_total > 0:
                delta_today_pct = 100.0
            else:
                delta_today_pct = 0.0

            current_screenings = session.execute(
                select(Screening).where(Screening.created_at >= range_start)
            ).scalars().all()

            total_curr = len(current_screenings)
            if total_curr > 0:
                avg_processing_time_ms = round(
                    sum(s.processing_time_ms for s in current_screenings) / total_curr, 1
                )
                average_risk_score = round(
                    sum(s.risk_score for s in current_screenings) / total_curr, 1
                )
                high_risk_events_count = sum(
                    1 for s in current_screenings
                    if s.risk_score >= 65
                    or s.risk_level == "HIGH_RISK"
                    or s.decision == "HIGH_RISK_REVIEW_REQUIRED"
                )
            else:
                avg_processing_time_ms = 0.0
                average_risk_score = 0.0
                high_risk_events_count = 0

            prior_times = session.execute(
                select(Screening.processing_time_ms).where(
                    (Screening.created_at >= prior_range_start) & (Screening.created_at < range_start)
                )
            ).scalars().all()
            if prior_times and len(prior_times) > 0:
                prior_avg = sum(prior_times) / len(prior_times)
                if prior_avg > 0:
                    avg_time_delta_pct = round(
                        ((avg_processing_time_ms - prior_avg) / prior_avg) * 100.0, 1
                    )
                else:
                    avg_time_delta_pct = 0.0
            else:
                avg_time_delta_pct = 0.0

            timeline = []
            for i in range(num_buckets):
                b_start = range_start + i * bucket_delta
                b_end = b_start + bucket_delta
                b_records = [
                    s for s in current_screenings
                    if b_start <= _normalize_utc(s.created_at) < b_end
                    or (i == num_buckets - 1 and b_start <= _normalize_utc(s.created_at) <= now)
                ]
                count = len(b_records)
                risk_avg = round(sum(s.risk_score for s in b_records) / count, 1) if count > 0 else 0.0
                timeline.append({
                    "timestamp": b_start.strftime(fmt),
                    "risk_score": risk_avg,
                    "count": count,
                })

            return {
                "range": range_key,
                "today_total": today_total,
                "yesterday_total": yesterday_total,
                "delta_today_pct": delta_today_pct,
                "avg_processing_time_ms": avg_processing_time_ms,
                "avg_time_delta_pct": avg_time_delta_pct,
                "average_risk_score": average_risk_score,
                "high_risk_events_count": high_risk_events_count,
                "timeline": timeline,
            }

    def recent_notifications(
        self,
        limit: int = 10,
        unread_only: bool = False,
    ) -> list[dict[str, Any]]:
        with self._database.session() as session:
            stmt = select(Screening).where(
                (Screening.risk_score >= 65) |
                (Screening.risk_level == "HIGH_RISK") |
                (Screening.decision == "HIGH_RISK_REVIEW_REQUIRED") |
                (Screening.status_color == "RED")
            ).order_by(Screening.id.desc())
            if unread_only:
                stmt = stmt.where(Screening.decision == "HIGH_RISK_REVIEW_REQUIRED")
            rows = session.execute(stmt.limit(limit)).scalars().all()
            notifications = []
            for s in rows:
                doc_str = s.request_id[:8] if s.request_id else "UNKNOWN"
                is_read = s.decision in {"CLEARED", "REVIEW"}
                notifications.append({
                    "id": f"notif-{s.id}",
                    "type": "HIGH_RISK_ALERT",
                    "message": f"High risk detected on screening {doc_str}",
                    "created_at": s.created_at.isoformat() if s.created_at else "",
                    "read": is_read,
                })
            return notifications

    @staticmethod
    def _count(session, condition) -> int:
        stmt = select(func.count()).select_from(Screening)
        if condition is not None:
            stmt = stmt.where(condition)
        return int(session.execute(stmt).scalar_one())


class AuditLogRepository:
    """Standalone audit events (e.g. failed persistence attempts)."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def record(
        self,
        *,
        screening_id: Optional[int],
        event_type: str,
        request_id: str,
        message: str,
        created_at: Optional[datetime.datetime] = None,
    ) -> AuditLog:
        with self._database.session() as session:
            event = AuditLog(
                screening_id=screening_id,
                event_type=event_type,
                created_at=created_at or utcnow_naive(),
                request_id=request_id,
                message=message,
            )
            session.add(event)
            session.commit()
            session.refresh(event)
            return event


class UserRepository:
    """Operator accounts; stores only hashed passwords."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def get_by_username(self, username: str) -> Optional[User]:
        identifier = username.strip()
        with self._database.session() as session:
            # 1. Exact match on username, officer_id, or email
            user = session.execute(
                select(User).where(
                    (User.username == identifier) |
                    (User.officer_id == identifier) |
                    (User.email == identifier)
                )
            ).scalars().first()
            if user is not None:
                return user

            # 2. Case-insensitive match on username, officer_id, or email
            user = session.execute(
                select(User).where(
                    (func.lower(User.username) == identifier.lower()) |
                    (func.lower(User.officer_id) == identifier.lower()) |
                    (func.lower(User.email) == identifier.lower())
                )
            ).scalars().first()
            if user is not None:
                return user

            # 3. Officer ID badge pattern, e.g. "LT-04" or "LT-4" -> user with id=4
            import re
            m = re.match(r"^(?:LT|OFFICER)-?0*(\d+)$", identifier, re.IGNORECASE)
            if m:
                uid = int(m.group(1))
                return session.get(User, uid)
            return None

    def get_by_email(self, email: str) -> Optional[User]:
        with self._database.session() as session:
            return session.execute(
                select(User).where(User.email == email)
            ).scalar_one_or_none()

    def get_by_id(self, user_id: int) -> Optional[User]:
        with self._database.session() as session:
            return session.get(User, user_id)

    def find_by_token_hash(self, token_hash: str) -> Tuple[Optional[AuthToken], Optional[User]]:
        with self._database.session() as session:
            token = session.execute(
                select(AuthToken).where(AuthToken.token_hash == token_hash)
            ).scalar_one_or_none()
            if token is None:
                return None, None
            user = session.get(User, token.user_id)
            return token, user

    def create(
        self,
        *,
        username: str,
        email: str,
        full_name: str,
        role: str,
        password_hash: str,
        officer_id: Optional[str] = None,
    ) -> User:
        with self._database.session() as session:
            user = User(
                username=username,
                email=email,
                full_name=full_name,
                role=role,
                officer_id=officer_id,
                password_hash=password_hash,
                is_active=True,
            )
            session.add(user)
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise PersistenceError("Username or email already exists.") from exc
            except SQLAlchemyError as exc:
                session.rollback()
                raise PersistenceError("Could not create user.") from exc
            session.refresh(user)
            return user


class AuthTokenRepository:
    """Bearer token persistence (only SHA-256 hashes are stored)."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def create(self, *, user_id: int, token_hash: str,
               expires_at: Optional[datetime.datetime]) -> AuthToken:
        with self._database.session() as session:
            token = AuthToken(
                token_hash=token_hash,
                user_id=user_id,
                expires_at=expires_at,
            )
            session.add(token)
            try:
                session.commit()
            except SQLAlchemyError as exc:
                session.rollback()
                raise PersistenceError("Could not create auth token.") from exc
            session.refresh(token)
            return token

    def delete_by_hash(self, token_hash: str) -> None:
        with self._database.session() as session:
            session.execute(delete(AuthToken).where(AuthToken.token_hash == token_hash))
            session.commit()
