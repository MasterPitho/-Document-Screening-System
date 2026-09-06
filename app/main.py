"""
Application factory, HTTP routes, and lifecycle for the Document Screening Engine.

The routes are module-level functions that read their singletons (settings,
model manager, validation limits, tampering detector, risk engine, database
and its repositories) from ``request.app.state``. No endpoint performs raw
database queries; all persistence goes through the repository layer so the API
surface stays thin and testable.

Startup is graceful: if PostgreSQL is temporarily unavailable, the app still
starts and ``/ready`` reports the connection state; the screening endpoint then
returns a controlled 503 instead of a stack trace.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.auth import (
    as_user_out,
    bootstrap_admin,
    extract_optional_user,
    get_current_user,
    verify_password,
    _generate_token,
    _hash_token,
    _hash_password,
)
from app.api.helpers import get_request_id, logger, structured_error
from app.config import Settings
from app.db.database import Database, DatabaseConnector, build_database, utcnow_naive
from app.db.models import Screening
from app.db.repositories import (
    AuditLogRepository,
    AuthTokenRepository,
    DuplicateRequestError,
    PersistenceError,
    ScreeningRepository,
    UserRepository,
)
from app.logging_setup import redact_mrz, set_up_logging
from app.models.schemas import (
    LivenessResultSchema,
    LoginRequest,
    LoginResponse,
    ReportSummary,
    RegisterRequest,
    ScreenResponse,
    ScreeningFactorOut,
    ScreeningListResponse,
    ScreeningRecordOut,
    ScreeningStats,
    UserOut,
)
from app.security.image_validation import ImageValidationLimits
from app.services import mrz as mrz_mod
from app.services import risk_engine as risk_mod
from app.services.face_recognition import ModelManager
from app.services.liveness import PassiveLivenessDetector
from app.services.risk_engine import RiskEngine
from app.services.tampering import TamperingDetector

import datetime


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app(
    settings: Optional[Settings] = None,
    model_manager: Optional[ModelManager] = None,
    database: Optional[Database] = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.validate()
    set_up_logging(settings.log_level, settings.api_env)

    database = database or build_database(
        settings.database_url, connect_timeout_s=settings.db_connect_timeout)
    model_manager = model_manager or ModelManager(settings)
    validation_limits = ImageValidationLimits(settings)
    risk_engine = RiskEngine(settings)
    tampering_detector = TamperingDetector(settings)
    liveness_detector = PassiveLivenessDetector(settings)

    # Repositories own every database session.
    screening_repo = ScreeningRepository(database)
    audit_repo = AuditLogRepository(database)
    user_repo = UserRepository(database)
    token_repo = AuthTokenRepository(database)

    # Development/test convenience: cheap schema ensure. In production the
    # Docker entrypoint runs `alembic upgrade head`; failures here (e.g. an
    # unreachable PostgreSQL) are ignored and surfaced via /ready.
    database_ready = database.create_all(fail_silently=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Do NOT assume the database is ready just because we started.
        ok, error = DatabaseConnector(database).initialize(
            attempts=max(1, settings.db_connect_retries),
            delay_s=settings.db_retry_delay,
        )
        app.state.database_ready = ok
        if not ok:
            logger.error("database_init_failed", type=error)
        else:
            try:
                bootstrap_admin(database, settings)
            except Exception:  # noqa: BLE001 - never crash startup for admin bootstrap
                logger.warning("admin_bootstrap_failed")
        # Model loading happens once at startup, never per request. A failure here
        # must not crash the API; readiness simply reports not_ready.
        model_manager.initialize_face_backend()
        liveness_detector.initialize()  # fails safe to the heuristic fallback
        yield

    app = FastAPI(
        title="Document Screening Engine",
        description=(
            "Border checkpoint document screening: ICAO 9303 TD3 MRZ validation, "
            "multi-signal image tampering analysis, ArcFace embedding face "
            "verification, a deterministic, explainable risk score, and a "
            "PostgreSQL-backed privacy-preserving audit trail. "
            "All signals are heuristic and always require human review."
        ),
        version="1.2.0",
        lifespan=lifespan,
    )

    app.state.settings = settings
    app.state.database = database
    app.state.database_ready = database_ready
    app.state.model_manager = model_manager
    app.state.validation = validation_limits
    app.state.risk_engine = risk_engine
    app.state.tampering = tampering_detector
    app.state.liveness = liveness_detector
    app.state.screening_repo = screening_repo
    app.state.audit_repo = audit_repo
    app.state.user_repo = user_repo
    app.state.token_repo = token_repo

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Request ID middleware: validate/echo X-Request-ID -----------------
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = get_request_id(request)
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # ---- Exception handlers: consistent, privacy-safe error bodies ---------
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "")
        logger.info("http_error", path=request.url.path, status=exc.status_code,
                    request_id=request_id)
        return structured_error(
            exc.status_code, str(exc.detail), detail=str(exc.detail),
            request_id=request_id,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", "")
        logger.info("validation_error", path=request.url.path, request_id=request_id)
        return structured_error(
            422, "Request validation failed.", detail=exc.errors(),
            request_id=request_id,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "")
        logger.error("unhandled_screening_error", path=request.url.path,
                     type=type(exc).__name__, request_id=request_id)
        return structured_error(
            500,
            "Document screening failed unexpectedly.",
            request_id=request_id,
        )

    _register_routes(app)
    return app


# --------------------------------------------------------------------------- #
# Route registration
# --------------------------------------------------------------------------- #
def _register_routes(app: FastAPI) -> None:
    # ---- Screening --------------------------------------------------------
    @app.post("/api/v1/screen", response_model=ScreenResponse)
    async def screen_document(
        request: Request,
        document_image: UploadFile = File(...),
        live_photo: Optional[UploadFile] = File(None),
        mrz_line1: Optional[str] = Form(None),
        mrz_line2: Optional[str] = Form(None),
        document_type: Optional[str] = Form(None),
    ) -> Dict[str, object]:
        state = request.app.state
        settings = state.settings
        validation_limits = state.validation
        tampering_detector = state.tampering
        risk_engine = state.risk_engine
        model_manager = state.model_manager
        liveness_detector = state.liveness
        screening_repo = state.screening_repo
        audit_repo = state.audit_repo

        request_id = getattr(request.state, "request_id", get_request_id(request))
        started_at = time.perf_counter()
        logger.info("screen_request_received", request_id=request_id,
                    endpoint="/api/v1/screen")
        if bool(mrz_line1) != bool(mrz_line2):
            raise HTTPException(
                status_code=400,
                detail="mrz_line1 and mrz_line2 must be provided together (both or neither).",
            )

        allowed_document_types = {"auto", "td3", "passport", "td1", "national_id", "aadhaar"}
        document_type = (document_type or "auto").strip().lower() or "auto"
        if document_type not in allowed_document_types:
            raise HTTPException(
                status_code=422,
                detail="document_type must be one of: "
                       "auto, td3/passport, td1/national_id/aadhaar.",
            )

        try:
            doc_bytes = await document_image.read(validation_limits.max_bytes + 1)
            validation_limits.validate(doc_bytes, document_image.content_type,
                                       "Document image", document_image.filename)
            live_bytes: Optional[bytes] = None
            if live_photo:
                live_bytes = await live_photo.read(validation_limits.max_bytes + 1)
                validation_limits.validate(live_bytes, live_photo.content_type,
                                           "Live photo", live_photo.filename)
        finally:
            await document_image.close()
            if live_photo:
                await live_photo.close()

        # 1. Tampering analysis (multi-signal, heuristic)
        try:
            tamper_result = tampering_detector.analyze(doc_bytes)
        except Exception:  # noqa: BLE001 - fail safe, never crash the pipeline
            tamper_result = {
                "status": "ERROR", "score": 0.0, "confidence": 0.0,
                "signals": {}, "suspicious_regions": [],
                "explanation": ["Tampering analysis failed; secondary inspection required."],
                "analysis_type": "multi-signal heuristic",
            }
        tamper_result["module_state"] = risk_mod.tampering_module_state(tamper_result)

        # 2. Face verification (ArcFace embeddings; NOT_AVAILABLE if no model)
        try:
            face_result = model_manager.face_engine.verify(
                doc_bytes, live_bytes, operator_name="insightface-arcface")
        except Exception:  # noqa: BLE001
            face_result = {
                "status": "ERROR", "similarity_score": None, "threshold": None,
                "matched": None,
                "explanation": "Face recognition failed internally; secondary inspection required.",
            }
        face_result["module_state"] = risk_mod.face_module_state(face_result)

        # 3. Passive liveness (PAD) on the live capture; in-memory only.
        try:
            liveness_detection = liveness_detector.analyze(live_bytes or b"")
        except Exception:  # noqa: BLE001 - fail safe
            liveness_detection = liveness_detector._not_checked(
                "Liveness analysis failed internally.")
        liveness_result = liveness_detection.to_dict()
        liveness_result["module_state"] = risk_mod.liveness_module_state(liveness_result)

        # 4. MRZ / document parse: submitted lines, otherwise the parser strategy.
        mrz_result = {}
        if mrz_line1 and mrz_line2:
            mrz_data = mrz_mod.parse_td3_mrz(
                mrz_line1, mrz_line2, year_pivot=settings.mrz_year_pivot)
            valid = mrz_data.get("status") == "VALID"
            mrz_result = {
                "detected": bool(valid),
                "source": "form",
                "status": str(mrz_data.get("status", "MALFORMED")),
                "confidence": 1.0 if valid else 0.5,
                "line1": redact_mrz(mrz_line1),
                "line2": redact_mrz(mrz_line2),
                "data": mrz_data,
                "format": "TD3",
                "document_type": "PASSPORT",
            }
        else:
            if document_type == "auto":
                mrz_result = mrz_mod.extract_mrz_from_image(doc_bytes, settings)
            else:
                mrz_result = mrz_mod.extract_mrz_from_image(
                    doc_bytes, settings, document_type=document_type)
        mrz_result["module_state"] = risk_mod.mrz_module_state(mrz_result)

        # 5. Deterministic, explainable risk decision.
        risk = risk_engine.evaluate(
            mrz_result=mrz_result,
            face_result=face_result,
            tamper_result=tamper_result,
            liveness_result=liveness_result,
            image_quality=1.0,
        )

        processing_time_ms = int((time.perf_counter() - started_at) * 1000)

        response: Dict[str, object] = {
            "status": "SCREENED",
            "request_id": request_id,
            "processing_time_ms": processing_time_ms,
            "document": {
                "format": str(mrz_result.get("format", "UNKNOWN")),
                "type": "PASSPORT" if mrz_result.get("detected") else "UNKNOWN",
                "document_type": str(mrz_result.get("document_type", "UNKNOWN")),
            },
            "modules": {
                "tampering_analysis": tamper_result,
                "face_verification": face_result,
                "liveness": liveness_result,
                "mrz_validation": mrz_result,
            },
            "risk_assessment": risk,
            "mrz": mrz_result,
            "tampering_analysis": tamper_result,
            "face_verification": face_result,
            "liveness": LivenessResultSchema.from_service(liveness_result),
        }

        # 5. Persist the screening: single transaction (commit on success,
        #    rollback on failure). Raw images, embeddings, MRZ text, and
        #    passport numbers are never stored.
        try:
            current_user = extract_optional_user(request)
            screening = screening_repo.create(
                request_id=request_id,
                processing_time_ms=processing_time_ms,
                document_type=str(response["document"].get("type", "UNKNOWN")),
                mrz_status=str(mrz_result.get("status", "NOT_DETECTED")),
                face_status=str(face_result.get("status", "NOT_AVAILABLE")),
                face_similarity=face_result.get("similarity_score"),
                tampering_status=str(tamper_result.get("status", "CLEAN")),
                tampering_score=tamper_result.get("score"),
                liveness_status=str(liveness_result.get("liveness_status", "NOT_CHECKED")),
                liveness_score=liveness_result.get("liveness_score"),
                risk_score=int(risk["score"]),
                risk_level=str(risk["level"]),
                decision=str(risk["decision"]),
                status_color=str(risk["status"]),
                module_states=dict(risk["module_statuses"]),
                factor_list=risk["factors"],
                mrz_source=str(mrz_result.get("source", "none")),
                user_id=current_user.id if current_user else None,
            )
            response["persistence"] = {"status": "stored", "screening_id": screening.id}
        except DuplicateRequestError:
            logger.warning("duplicate_request_id", request_id=request_id)
            raise HTTPException(status_code=409,
                                detail="A screening with this request_id already exists.")
        except PersistenceError:
            # Never surface a database stack trace. The analysis itself is
            # valid, but it could not be persisted, so we must not present it
            # as a confirmed, recorded result.
            logger.error("screening_persistence_failed", request_id=request_id)
            try:
                audit_repo.record(screening_id=None, event_type="persistence.failed",
                                  request_id=request_id,
                                  message="screening result could not be persisted")
            except Exception:  # noqa: BLE001
                pass
            raise HTTPException(
                status_code=503,
                detail="Screening analysis completed but the result could not be "
                       "persisted; please retry or perform manual inspection.",
            )

        logger.info("screen_request_completed", request_id=request_id,
                    risk_level=risk["level"], risk_score=risk["score"],
                    processing_time_ms=processing_time_ms,
                    decision=risk["decision"])
        return response

    # ---- Auth -------------------------------------------------------------
    @app.post("/api/v1/auth/register", response_model=UserOut, status_code=201)
    def register(request: Request, payload: RegisterRequest) -> UserOut:
        username = payload.username.strip()
        email = payload.email.strip().lower()
        if not username or not email:
            raise HTTPException(status_code=422, detail="username and email are required.")
        try:
            user = request.app.state.user_repo.create(
                username=username,
                email=email,
                full_name=payload.full_name.strip(),
                role="officer",
                password_hash=_hash_password(payload.password),
            )
        except PersistenceError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return as_user_out(user)

    @app.post("/api/v1/auth/login", response_model=LoginResponse)
    def login(request: Request, payload: LoginRequest) -> LoginResponse:
        user = request.app.state.user_repo.get_by_username(payload.username.strip())
        if user is None or not user.is_active or not verify_password(
            payload.password, user.password_hash
        ):
            raise HTTPException(status_code=401, detail="Invalid credentials.")
        token = _generate_token()
        request.app.state.token_repo.create(
            user_id=user.id,
            token_hash=_hash_token(token),
            expires_at=utcnow_naive() + datetime.timedelta(
                hours=request.app.state.settings.auth_token_ttl_hours),
        )
        logger.info("auth_login", user_id=user.id)
        return LoginResponse(token=token, user=as_user_out(user))

    @app.post("/api/v1/auth/logout")
    def logout(request: Request) -> Dict[str, str]:
        authorization = request.headers.get("Authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            request.app.state.token_repo.delete_by_hash(_hash_token(token))
        return {"status": "ok"}

    @app.get("/api/v1/auth/me", response_model=UserOut)
    def me(current_user=Depends(get_current_user)) -> UserOut:
        return as_user_out(current_user)

    # ---- History & reporting (bearer token required) ----------------------
    @app.get("/api/v1/screenings", response_model=ScreeningListResponse)
    def list_screenings(
        request: Request,
        _current_user=Depends(get_current_user),
        limit: int = Query(20, ge=1, le=100),
        offset: int = Query(0, ge=0),
        decision: Optional[str] = Query(None),
        risk_level: Optional[str] = Query(None),
        date_from: Optional[datetime.datetime] = Query(None),
        date_to: Optional[datetime.datetime] = Query(None),
    ) -> ScreeningListResponse:
        repo: ScreeningRepository = request.app.state.screening_repo
        total, records = repo.list(
            decision=decision, risk_level=risk_level,
            date_from=date_from, date_to=date_to, limit=limit, offset=offset,
        )
        return ScreeningListResponse(
            total=total,
            limit=limit,
            offset=offset,
            decision=decision,
            risk_level=risk_level,
            date_from=date_from.isoformat() if date_from else None,
            date_to=date_to.isoformat() if date_to else None,
            records=[_as_screening_out(r) for r in records],
        )

    @app.get("/api/v1/screenings/{item_id}", response_model=ScreeningRecordOut)
    def get_screening(
        request: Request,
        item_id: str,
        _current_user=Depends(get_current_user),
    ) -> ScreeningRecordOut:
        record = _resolve_screening(request.app.state.screening_repo, item_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Screening record not found.")
        return _as_screening_out(record)

    @app.get("/api/v1/screenings/{item_id}/factors", response_model=list[ScreeningFactorOut])
    def get_screening_factors(
        request: Request,
        item_id: str,
        _current_user=Depends(get_current_user),
    ):
        repo: ScreeningRepository = request.app.state.screening_repo
        record = _resolve_screening(repo, item_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Screening record not found.")
        return [
            ScreeningFactorOut(
                id=f.id, factor_name=f.factor_name, severity=f.severity,
                weight=f.weight, description=f.description,
            )
            for f in repo.list_factors(record.id)
        ]

    @app.get("/api/v1/stats", response_model=ScreeningStats)
    def dashboard_stats(
        request: Request,
        _current_user=Depends(get_current_user),
    ) -> ScreeningStats:
        return ScreeningStats(**request.app.state.screening_repo.stats())

    @app.get("/api/v1/report/summary", response_model=ReportSummary)
    def report_summary(
        request: Request,
        _current_user=Depends(get_current_user),
    ) -> ReportSummary:
        summary = request.app.state.screening_repo.summary()
        return ReportSummary(
            total_screenings=summary["total"],
            cleared=summary["cleared"],
            secondary_inspection=summary["secondary_inspection"],
            high_risk=summary["high_risk"],
            avg_processing_time_ms=summary["avg_processing_time_ms"],
            by_decision=summary["by_decision"],
            by_risk_level=summary["by_risk_level"],
        )

    # ---- System -----------------------------------------------------------
    @app.get("/health")
    def health_check(request: Request) -> Dict[str, str]:
        # No sensitive information is exposed here.
        return {"status": "ok", "service": "Document Screening Engine",
                "env": request.app.state.settings.api_env}

    @app.get("/ready")
    def readiness_check(request: Request) -> JSONResponse:
        database: Database = request.app.state.database
        db_ok = database.ping()
        readiness = request.app.state.model_manager.readiness()
        modules: Dict[str, Any] = dict(readiness)
        modules["database"] = db_ok
        modules.update(request.app.state.liveness.readiness())
        ready = db_ok and bool(readiness.get("face_recognition"))
        return JSONResponse(status_code=200 if ready else 503, content={
            "status": "ready" if ready else "not_ready",
            "modules": modules,
        })


def _resolve_screening(repo: ScreeningRepository, item_id: str) -> Optional[Screening]:
    """Resolve a path segment that may be an integer id or a 32-char request id."""
    try:
        return repo.get(int(item_id))
    except ValueError:
        return repo.get_by_request_id(item_id)


def _as_screening_out(record: Screening) -> ScreeningRecordOut:
    created = record.created_at
    return ScreeningRecordOut(
        id=record.id,
        request_id=record.request_id,
        risk_score=record.risk_score,
        risk_level=record.risk_level,
        decision=record.decision,
        status_color=record.status_color,
        module_states=dict(record.module_states or {}),
        factors=list(record.factors or []),
        processing_time_ms=record.processing_time_ms,
        document_type=record.document_type,
        mrz_status=record.mrz_status,
        face_status=record.face_status,
        face_similarity=record.face_similarity,
        tampering_status=record.tampering_status,
        tampering_score=record.tampering_score,
        liveness_status=getattr(record, "liveness_status", None) or "NOT_CHECKED",
        liveness_score=getattr(record, "liveness_score", None),
        mrz_source=record.mrz_source,
        user_id=record.user_id,
        created_at=created.isoformat() if created else "",
    )


# Module-level ASGI app: builds once, uvicorn will run its lifespan at startup.
app = create_app()
