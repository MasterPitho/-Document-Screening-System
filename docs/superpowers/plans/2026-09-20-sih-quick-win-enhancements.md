# SIH Quick-Win Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add seven additive backend capabilities to the existing screening engine — hash-chain ledger, real watchlists with match integration, real notifications, CSV export, PDF document upload, and ICAO 9303 TD2 visa parsing — without touching the deployed screening pipeline's existing behaviour.

**Architecture:** Data-layer first (new SQLAlchemy tables + repositories following the existing repository pattern), then read-only/additive API endpoints mirroring existing `/api/v1/*` routes, then surgical, non-breaking integration points in `_screen_document_impl` (watchlist factor injection via a new optional `extra_factors` kwarg on `RiskEngine.evaluate`, best-effort ledger append + high-risk notification insert after screening persist, PDF-first-page render before image validation). Everything additive; no table changes to existing models; no column/migration changes.

**Tech Stack:** FastAPI + SQLAlchemy 2.x (existing), `pymupdf` (new dependency, pure wheel, no poppler), stdlib `hashlib`/`json`/`csv`/`io`, existing test stack (pytest + Starlette TestClient + DummyBackend face backend in `tests/conftest.py`).

**Spec:** `docs/superpowers/specs/2026-09-20-sih-quick-win-enhancements-design.md` (travels with this plan; the plan argues from the spec — executors read both).

## Global Constraints

- **NO `git commit` OR `git push` anywhere in this plan.** The user explicitly ordered: *"isko abhi kahi v push ya commit nahi karna hai"*. Every task's final step is a test run + a code-review checkpoint, never a commit. At the very end, show `git status` and ask the user before any commit.
- Python is **`C:\Python314\python`** on the worker. Always run tests with `C:\Python314\python -m pytest`.
- New dependency `pymupdf>=1.24` must be installed locally before Task 9 tests: `C:\Python314\python -m pip install "pymupdf>=1.24"`. Run this AFTER editing `requirements.txt` so the file and env stay in sync.
- Do NOT fix existing lint items in `app/main.py` (pre-existing F401/F841 on HEAD). New code must be ruff-clean (`ruff check app tests`).
- Follow existing patterns exactly: repositories own their sessions (open/commit/rollback/close inside one method); endpoints fetch repos from `request.app.state`; auth via `Depends(get_current_user)` / `Depends(require_role("officer", "supervisor"))`; naive-UTC datetimes via `app.db.database.utcnow_naive`.
- **Privacy contract (unchanged, strict):** never persist names, raw MRZ text, or document numbers in `Screenings`, `ledger_entries.payload`, or CSV export. Watchlist names live only in the `watchlist_entries` table and transient HTTP responses.
- Response shapes must remain backward compatible: `NotificationItem` and `WatchlistItem` fields are NOT renamed or removed (new read-only fields may be added). GET `/api/v1/notifications` and GET `/api/v1/watchlists` keep returning the same JSON shape.
- Tests must never depend on tesseract OCR. All integration tests drive the deterministic paths: `mrz_line1`+`mrz_line2` form fields (bypass OCR), and repository/service-level calls.
- Existing test suite must stay green: `C:\Python314\python -m pytest -q` (currently 248 passed).

## File Structure

New files:
- `app/services/watchlist.py` — document-number normalization + match helper.
- `app/services/ledger.py` — canonical JSON, SHA-256, hash-chain `LedgerRepository`.
- `app/services/pdf.py` — PDF detection + pymupdf first-page render.
- `app/db/models.py` — MODIFY (add `WatchlistEntry`, `Notification`, `LedgerEntry`).
- `app/db/repositories.py` — MODIFY (add `WatchlistRepository`, `NotificationRepository`, `ScreeningRepository.export_rows`).
- `app/services/mrz.py` — MODIFY (add `parse_td2_visa`, `VisaTD2Parser`, aliases, router registration).
- `app/services/risk_engine.py` — MODIFY (add optional `extra_factors` kwarg).
- `app/config.py` — MODIFY (add `max_pdf_bytes`, `pdf_render_max_dimension`, `ON_WATCHLIST` risk weight).
- `app/models/schemas.py` — MODIFY (add `WatchlistCreate`, `WatchlistUpdate`).
- `app/main.py` — MODIFY (repo construction + state, startup seeds, watchlist/notification/ledger endpoints, PDF injection, `risk_engine.evaluate(extra_factors=...)`, ledger + notification wiring after persist, `allowed_document_types` + `converted_from_pdf` field).
- `requirements.txt` — MODIFY (add `pymupdf>=1.24`).

Tests (all new):
- `tests/test_quickwin_config.py`, `tests/test_watchlist.py`, `tests/test_watchlist_api.py`, `tests/test_watchlist_integration.py`, `tests/test_notifications.py`, `tests/test_ledger.py`, `tests/test_ledger_api.py`, `tests/test_csv_export.py`, `tests/test_pdf_upload.py`, `tests/test_visa_td2.py`.

---

### Task 1: Config extensions (PDF limits + watchlist risk weight)

**Files:**
- Modify: `app/config.py` (fields, `from_env`, `validate`), `app/models/schemas.py` (nothing yet)
- Test: `tests/test_quickwin_config.py`

**Interfaces:**
- Consumes: existing `Settings` frozen dataclass + `_int_value`.
- Produces: `settings.max_pdf_bytes: int`, `settings.pdf_render_max_dimension: int`, `settings.risk_weights["ON_WATCHLIST"]: int`; `Settings.validate()` accepts the new required factor.

- [ ] **Step 1: Write the failing test** — `tests/test_quickwin_config.py`:

```python
from app.config import Settings


def test_pdf_settings_have_defaults():
    settings = Settings.from_env()
    assert settings.max_pdf_bytes == 20 * 1024 * 1024
    assert settings.pdf_render_max_dimension == 4096


def test_on_watchlist_weight_configured():
    settings = Settings.from_env()
    assert settings.risk_weights["ON_WATCHLIST"] == 40


def test_validate_accepts_new_factors():
    Settings.from_env().validate()  # must not raise
```

- [ ] **Step 2: Run to verify it fails**

Run: `C:\Python314\python -m pytest tests/test_quickwin_config.py -v`
Expected: FAIL (`AttributeError: 'Settings' object has no attribute 'max_pdf_bytes'`).

- [ ] **Step 3: Implement**

In `app/config.py`:
- Add fields after `max_image_height`: `max_pdf_bytes: int` and `pdf_render_max_dimension: int`.
- In `from_env`'s `default_weights` dict add:
  `"ON_WATCHLIST": _int_value("RISK_ON_WATCHLIST", 40),`
- In the `return cls(...)` add:
  ```python
  max_pdf_bytes=_int_value("MAX_PDF_BYTES", 20 * 1024 * 1024),
  pdf_render_max_dimension=_int_value("PDF_RENDER_MAX_DIMENSION", 4096),
  ```
- In `validate()`, after the `max_image_width/height` check:
  ```python
  _check(self.max_pdf_bytes > 0, "MAX_PDF_BYTES must be greater than zero")
  _check(self.pdf_render_max_dimension > 0, "PDF_RENDER_MAX_DIMENSION must be greater than zero")
  ```
- Add `"ON_WATCHLIST"` to the `required_factors` set.

- [ ] **Step 4: Run to verify it passes**

Run: `C:\Python314\python -m pytest tests/test_quickwin_config.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Review checkpoint** — no commit (global constraint).

---

### Task 2: Watchlist data layer (table + repository + seed)

**Files:**
- Create: `app/services/watchlist.py`
- Modify: `app/db/models.py` (append `WatchlistEntry` class), `app/db/repositories.py` (append `WatchlistRepository` + severity validation)
- Test: `tests/test_watchlist.py`

**Interfaces:**
- Consumes: `Database` from `app.db.database`, `utcnow_naive`, existing model/SQLAlchemy imports in `repositories.py`.
- Produces:
  - `app.services.watchlist.normalize_document_number(value: str) -> str`
  - `WatchlistRepository(database)` with methods:
    - `create(*, name: str, document_number: str, reason: str = "", severity: str = "MEDIUM", is_demo_data: bool = False, source: str = "MANUAL_ENTRY") -> WatchlistEntry`
    - `update(entry_id: int, *, name=None, document_number=None, reason=None, severity=None) -> Optional[WatchlistEntry]`
    - `delete(entry_id: int) -> bool`
    - `get(entry_id: int) -> Optional[WatchlistEntry]`
    - `list(*, limit: int = 100, offset: int = 0) -> Tuple[int, list[WatchlistEntry]]`
    - `search(term: str) -> Tuple[int, list[WatchlistEntry]]`
    - `match(document_number: str) -> Optional[WatchlistEntry]`
    - `seed_defaults() -> int`

- [ ] **Step 1: Write the failing test** — `tests/test_watchlist.py`:

```python
import pytest

from app.db.models import WatchlistEntry
from app.services.watchlist import normalize_document_number


def test_normalize_removes_separators_and_uppercases():
    assert normalize_document_number("L-898 902c") == "L898902C"


def test_create_and_get():
    WatchlistEntry  # ensure model importable
```

Write the full repository tests after seeing the class shape; the core tests are:

```python
def test_create_get_match(app_and_client, tmp_path):
    from app.db.database import Database, build_database
    from app.db.repositories import WatchlistRepository
    db = build_database(f"sqlite:///{(tmp_path / 'wl.db').as_posix()}")
    repo = WatchlistRepository(db)
    entry = repo.create(name="Subject A", document_number="L-898 902c",
                        reason="Lookout", severity="HIGH")
    assert entry.id is not None
    assert entry.document_number_normalized == "L898902C"
    match = repo.match("l898902c")
    assert match is not None and match.id == entry.id
    assert repo.match("X1234567") is None
```

Provide the remaining cases (update-only-provided-fields, delete-missing-false, duplicate-create-allowed, seed-once-only, seed-list-is-demo) as concrete functions following the same `repo = WatchlistRepository(build_database(...))` setup.

- [ ] **Step 2: Run to verify it fails**

Run: `C:\Python314\python -m pytest tests/test_watchlist.py -v`
Expected: FAIL (`No module named 'app.services.watchlist'` / import errors).

- [ ] **Step 3: Implement**

`app/services/watchlist.py`:

```python
"""Watchlist document-number normalization helpers."""

from __future__ import annotations

import re

_NON_ALPHANUMERIC = re.compile(r"[^A-Z0-9]")


def normalize_document_number(value: str) -> str:
    """Normalize a document number for exact matching (case + separators)."""
    return _NON_ALPHANUMERIC.sub("", (value or "").upper())
```

`app/db/models.py` — append before `ScreeningFactor`:

```python
class WatchlistEntry(Base):
    """Manually curated watchlist records; matched against screened numbers."""

    __tablename__ = "watchlist_entries"

    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    document_number = Column(String(64), nullable=False)
    document_number_normalized = Column(String(64), nullable=False, index=True)
    reason = Column(Text, nullable=False, default="")
    severity = Column(String(10), nullable=False, default="MEDIUM")  # LOW|MEDIUM|HIGH|CRITICAL
    is_demo_data = Column(Boolean, nullable=False, default=False)
    source = Column(String(50), nullable=False, default="MANUAL_ENTRY")
    created_at = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_watchlist_doc_number_normalized", document_number_normalized),
    )
```

`app/db/repositories.py` — append at end:

```python
class WatchlistRepository:
    """CRUD + normalized matching for watchlist records."""

    VALID_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    def __init__(self, database: Database) -> None:
        self._database = database

    @staticmethod
    def _require_severity(severity: str) -> None:
        if severity not in WatchlistRepository.VALID_SEVERITIES:
            raise ValueError(
                f"severity must be one of {sorted(WatchlistRepository.VALID_SEVERITIES)}")

    def create(self, *, name: str, document_number: str, reason: str = "",
               severity: str = "MEDIUM", is_demo_data: bool = False,
               source: str = "MANUAL_ENTRY") -> WatchlistEntry:
        self._require_severity(severity)
        with self._database.session() as session:
            entry = WatchlistEntry(
                name=name,
                document_number=document_number,
                document_number_normalized=normalize_document_number(document_number),
                reason=reason,
                severity=severity,
                is_demo_data=is_demo_data,
                source=source,
                created_at=utcnow_naive(),
            )
            session.add(entry)
            session.commit()
            session.refresh(entry)
            return entry

    def update(self, entry_id: int, *, name: Optional[str] = None,
               document_number: Optional[str] = None, reason: Optional[str] = None,
               severity: Optional[str] = None) -> Optional[WatchlistEntry]:
        if severity is not None:
            self._require_severity(severity)
        with self._database.session() as session:
            entry = session.get(WatchlistEntry, entry_id)
            if entry is None:
                return None
            if name is not None:
                entry.name = name
            if document_number is not None:
                entry.document_number = document_number
                entry.document_number_normalized = normalize_document_number(document_number)
            if reason is not None:
                entry.reason = reason
            if severity is not None:
                entry.severity = severity
            session.commit()
            session.refresh(entry)
            return entry

    def delete(self, entry_id: int) -> bool:
        with self._database.session() as session:
            entry = session.get(WatchlistEntry, entry_id)
            if entry is None:
                return False
            session.delete(entry)
            session.commit()
            return True

    def get(self, entry_id: int) -> Optional[WatchlistEntry]:
        with self._database.session() as session:
            return session.get(WatchlistEntry, entry_id)

    def list(self, *, limit: int = 100, offset: int = 0) -> Tuple[int, list[WatchlistEntry]]:
        with self._database.session() as session:
            total = int(session.execute(
                select(func.count()).select_from(WatchlistEntry)).scalar_one())
            rows = session.execute(
                select(WatchlistEntry).order_by(WatchlistEntry.id).limit(limit).offset(offset)
            ).scalars().all()
            return total, list(rows)

    def search(self, term: str) -> Tuple[int, list[WatchlistEntry]]:
        like = f"%{(term or '').strip()}%"
        with self._database.session() as session:
            stmt = select(WatchlistEntry).where(
                WatchlistEntry.name.ilike(like) |
                WatchlistEntry.document_number.ilike(like) |
                WatchlistEntry.reason.ilike(like)
            ).order_by(WatchlistEntry.id)
            rows = session.execute(stmt).scalars().all()
            return len(rows), list(rows)

    def match(self, document_number: str) -> Optional[WatchlistEntry]:
        normalized = normalize_document_number(document_number)
        with self._database.session() as session:
            return session.execute(
                select(WatchlistEntry).where(
                    WatchlistEntry.document_number_normalized == normalized)
            ).scalars().first()

    def seed_defaults(self) -> int:
        """Insert the three demo entries ONLY if the table is empty."""
        with self._database.session() as session:
            if session.execute(select(func.count()).select_from(WatchlistEntry)).scalar_one() > 0:
                return 0
            demos = [
                ("Tehran Ali", "V-449021", "Lookout circular", "HIGH"),
                ("Marcus Vance", "P8831042", "Interpol Red Notice alert", "CRITICAL"),
                ("Elena Rostova", "E7719203", "Stolen blank passport registry", "HIGH"),
            ]
            for name, doc_number, reason, severity in demos:
                session.add(WatchlistEntry(
                    name=name, document_number=doc_number,
                    document_number_normalized=normalize_document_number(doc_number),
                    reason=reason, severity=severity,
                    is_demo_data=True,
                    source="DEMO_DATA_NOT_FOR_OPERATIONAL_USE", created_at=utcnow_naive(),
                ))
            session.commit()
            return len(demos)
```

Add to the existing imports in `repositories.py`: `from app.db.models import ..., WatchlistEntry` and `from app.services.watchlist import normalize_document_number`. (Confirm `Tuple` and `Optional` are already imported — they are, per the current header.)

- [ ] **Step 4: Run to verify it passes**

Run: `C:\Python314\python -m pytest tests/test_watchlist.py -v`
Expected: all PASS.

- [ ] **Step 5: Review checkpoint** — no commit (global constraint).

---

### Task 3: Watchlist API (replace hardcoded stub with real CRUD)

**Files:**
- Modify: `app/models/schemas.py` (add `WatchlistCreate`, `WatchlistUpdate`), `app/main.py` (repo construction + `state.watchlist_repo`, replace lines 849-886 endpoint, add POST/PATCH/DELETE)
- Test: `tests/test_watchlist_api.py`

**Interfaces:**
- Consumes: `WatchlistRepository` from Task 2; `WatchlistItem` schema; `get_current_user`/`require_role`.
- Produces: `GET|POST /api/v1/watchlists`, `PATCH|DELETE /api/v1/watchlists/{entry_id}`, and helper `_to_watchlist_item(entry) -> WatchlistItem`.

- [ ] **Step 1: Write the failing test** — `tests/test_watchlist_api.py`:

```python
def _seed(client):
    token = client.post("/api/v1/auth/register", json={
        "username": "wl_officer", "email": "wl@example.com",
        "full_name": "WL Officer", "password": "Password123!",
    }).json()["username"]
    token = client.post("/api/v1/auth/login", json={
        "username": "wl_officer", "password": "Password123!",
    }).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_list_returns_seeded_demo_entries(app_and_client):
    _, client = app_and_client
    headers = _seed(client)
    resp = client.get("/api/v1/watchlists", headers=headers)
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) > 0
    assert all(item["is_demo_data"] for item in items)
    assert items[0]["source"] == "DEMO_DATA_NOT_FOR_OPERATIONAL_USE"


def test_create_list_patch_delete_flow(app_and_client):
    _, client = app_and_client
    headers = _seed(client)
    created = client.post("/api/v1/watchlists", headers=headers, json={
        "name": "New Subject", "document_number": "L-898 902c",
        "reason": "Vigilance case", "severity": "HIGH",
    })
    assert created.status_code == 201
    body = created.json()
    assert body["is_demo_data"] is False
    assert body["source"] == "MANUAL_ENTRY"
    entry_id = body["id"]

    patched = client.patch(f"/api/v1/watchlists/{entry_id}", headers=headers,
                           json={"reason": "Updated rationale"})
    assert patched.status_code == 200
    assert patched.json()["reason"] == "Updated rationale"
    assert patched.json()["name"] == "New Subject"

    deleted = client.delete(f"/api/v1/watchlists/{entry_id}", headers=headers)
    assert deleted.status_code == 204
    assert client.delete(f"/api/v1/watchlists/{entry_id}", headers=headers).status_code == 404


def test_create_validation_and_auth(app_and_client):
    _, client = app_and_client
    assert client.get("/api/v1/watchlists").status_code == 401
    headers = _seed(client)
    bad = client.post("/api/v1/watchlists", headers=headers, json={
        "name": "X", "document_number": "L898902C", "severity": "EXTREME"})
    assert bad.status_code == 422
    short = client.post("/api/v1/watchlists", headers=headers, json={
        "name": "OK", "document_number": "AB", "severity": "LOW"})
    assert short.status_code == 422
```

- [ ] **Step 2: Run to verify it fails**

Run: `C:\Python314\python -m pytest tests/test_watchlist_api.py -v`
Expected: FAIL (`POST /api/v1/watchlists` returns 405, `is_demo_data` back-compat list still hardcoded).

- [ ] **Step 3: Implement**

`app/models/schemas.py` — add imports `Literal` (typing) then:

```python
class WatchlistCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    document_number: str = Field(min_length=3, max_length=64)
    reason: str = Field(default="", max_length=500)
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "MEDIUM"


class WatchlistUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=120)
    document_number: Optional[str] = Field(default=None, min_length=3, max_length=64)
    reason: Optional[str] = Field(default=None, max_length=500)
    severity: Optional[Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]] = None
```

`app/main.py`:
- In the repo-construction block (after `token_repo = AuthTokenRepository(database)` at line ~224):
  ```python
  watchlist_repo = WatchlistRepository(database)
  ledger_repo = LedgerRepository(database)   # arrives in Task 6; import then
  notification_repo = NotificationRepository(database)  # arrives in Task 5; import then
  ```
  (For THIS task only wire `watchlist_repo`; Tasks 5/6/7 add their own lines. Set `app.state.watchlist_repo = watchlist_repo` next to the other `app.state.*` assignments at line ~274. Call `watchlist_repo.seed_defaults()` right after the schema-ensure block so a fresh DB always returns demo rows.)
- Add import at top: `from app.db.repositories import WatchlistRepository` and `from app.models.schemas import ..., WatchlistCreate, WatchlistUpdate`.
- Add helper module-level function:

```python
def _to_watchlist_item(entry) -> WatchlistItem:
    return WatchlistItem(
        id=entry.id,
        name=entry.name,
        document_number=entry.document_number,
        reason=entry.reason,
        severity=entry.severity,
        created_at=entry.created_at.isoformat() if entry.created_at else "",
        is_demo_data=bool(entry.is_demo_data),
        source=entry.source,
    )
```

- Replace the whole `get_watchlists` block (lines 849-886) with:

```python
    @app.get("/api/v1/watchlists", response_model=list[WatchlistItem])
    def get_watchlists(
        request: Request,
        search: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
        _current_user=Depends(get_current_user),
    ) -> list[WatchlistItem]:
        repo: WatchlistRepository = request.app.state.watchlist_repo
        if search:
            _, rows = repo.search(search)
        else:
            _, rows = repo.list(limit=limit, offset=offset)
        return [_to_watchlist_item(entry) for entry in rows]

    @app.post("/api/v1/watchlists", response_model=WatchlistItem, status_code=201)
    def create_watchlist(
        request: Request,
        body: WatchlistCreate,
        _current_user=Depends(require_role("officer", "supervisor")),
    ) -> WatchlistItem:
        repo: WatchlistRepository = request.app.state.watchlist_repo
        entry = repo.create(
            name=body.name.strip(),
            document_number=body.document_number.strip(),
            reason=body.reason.strip(),
            severity=body.severity,
        )
        return _to_watchlist_item(entry)

    @app.patch("/api/v1/watchlists/{entry_id}", response_model=WatchlistItem)
    def update_watchlist(
        request: Request,
        entry_id: int,
        body: WatchlistUpdate,
        _current_user=Depends(require_role("officer", "supervisor")),
    ) -> WatchlistItem:
        repo: WatchlistRepository = request.app.state.watchlist_repo
        entry = repo.update(
            entry_id,
            name=body.name.strip() if body.name else None,
            document_number=body.document_number.strip() if body.document_number else None,
            reason=body.reason.strip() if body.reason else None,
            severity=body.severity,
        )
        if entry is None:
            raise HTTPException(status_code=404, detail="Watchlist entry not found.")
        return _to_watchlist_item(entry)

    @app.delete("/api/v1/watchlists/{entry_id}", status_code=204)
    def delete_watchlist(
        request: Request,
        entry_id: int,
        _current_user=Depends(require_role("officer", "supervisor")),
    ) -> None:
        repo: WatchlistRepository = request.app.state.watchlist_repo
        if not repo.delete(entry_id):
            raise HTTPException(status_code=404, detail="Watchlist entry not found.")
```

- [ ] **Step 4: Run to verify it passes**

Run: `C:\Python314\python -m pytest tests/test_watchlist_api.py -v`
Expected: all PASS.

- [ ] **Step 5: Review checkpoint** — no commit (global constraint).

---

### Task 4: Watchlist match integration (risk factor injection)

**Files:**
- Modify: `app/services/risk_engine.py` (add optional `extra_factors` kwarg), `app/main.py` (`_screen_document_impl`: match lookup + extra factor + `watchlist` response field)
- Test: `tests/test_watchlist_integration.py`

**Interfaces:**
- Consumes: `WatchlistRepository.match` (Task 2); `RiskEngine.evaluate` (now with `extra_factors`).
- Produces: `RiskEngine.evaluate(..., extra_factors: Optional[List[Dict[str, Any]]] = None)`; screening response gains top-level key `watchlist` (`None` or `{"match": True, "watchlist_id": int, "severity": str}`).

- [ ] **Step 1: Write the failing test** — `tests/test_watchlist_integration.py`:

```python
import json

import numpy as np

from app.config import Settings
from app.services.risk_engine import RiskEngine, UnknownRiskFactorError

VALID_LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
UNEXPIRED_LINE2 = "L898902C36UTO7408122F3501014ZE184226B<<<<<16"

GREEN = {"status": "CLEAN", "score": 0.0}
FACE = {"status": "MATCH", "similarity_score": 0.9}
MRZ_OK = {"status": "VALID", "detected": True,
          "data": {"checks": {}, "is_expired": False}}
LIVE = {"liveness_status": "LIVE"}


def test_extra_factors_add_to_score():
    engine = RiskEngine(Settings.from_env())
    result = engine.evaluate(
        mrz_result=MRZ_OK, face_result=FACE, tamper_result=GREEN,
        liveness_result=LIVE,
        extra_factors=[{"factor": "ON_WATCHLIST", "detail": "x"}],
    )
    assert result["score"] == 40
    assert any(f["factor"] == "ON_WATCHLIST" for f in result["factors"])
    assert result["decision"] != "CLEARED"


def test_extra_factors_unknown_raises():
    engine = RiskEngine(Settings.from_env())
    import pytest
    with pytest.raises(UnknownRiskFactorError):
        engine.evaluate(
            mrz_result=MRZ_OK, face_result=FACE, tamper_result=GREEN,
            liveness_result=LIVE,
            extra_factors=[{"factor": "NOT_A_REAL_FACTOR", "detail": "x"}],
        )


def test_screen_matching_watchlist_adds_factor(app_and_client):
    _, client = app_and_client
    token = client.post("/api/v1/auth/register", json={
        "username": "wl_int", "email": "wlint@example.com",
        "full_name": "WL Int", "password": "Password123!",
    }).json()
    token = client.post("/api/v1/auth/login", json={
        "username": "wl_int", "password": "Password123!",
    }).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    created = client.post("/api/v1/watchlists", headers=headers, json={
        "name": "Test Subject", "document_number": "L898902C",
        "reason": "Integration test", "severity": "HIGH",
    })
    assert created.status_code == 201

    resp = client.post(
        "/api/v1/screen",
        files={"document_image": ("passport.jpg", _jpeg(), "image/jpeg")},
        data={"mrz_line1": VALID_LINE1, "mrz_line2": UNEXPIRED_LINE2},
        headers=headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    factors = body["risk_assessment"]["factors"]
    assert any(f["factor"] == "ON_WATCHLIST" for f in factors)
    assert body["watchlist"] == {"match": True, "watchlist_id": created.json()["id"],
                                 "severity": "HIGH"}
    # Privacy: the subject's NAME never appears anywhere in the response/persistence.
    assert "Test Subject" not in json.dumps(body)


def test_screen_non_matching_has_no_watchlist(app_and_client):
    _, client = app_and_client
    token = client.post("/api/v1/auth/login", json={
        "username": "wl_int", "password": "Password123!",
    }).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    resp = client.post(
        "/api/v1/screen",
        files={"document_image": ("passport.jpg", _jpeg(), "image/jpeg")},
        data={"mrz_line1": VALID_LINE1, "mrz_line2": UNEXPIRED_LINE2},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["watchlist"] is None
```

(`_jpeg` helper: white 800x500 JPEG via `PIL.Image`, exactly as in `tests/test_dashboard_extensions.py:22-25`. The full file must include that helper.)

- [ ] **Step 2: Run to verify it fails**

Run: `C:\Python314\python -m pytest tests/test_watchlist_integration.py -v`
Expected: FAIL (`TypeError: evaluate() got an unexpected keyword argument 'extra_factors'` and `body["watchlist"]` KeyError).

- [ ] **Step 3: Implement**

`app/services/risk_engine.py`:
- Change `evaluate` signature to add `extra_factors: Optional[List[Dict[str, Any]]] = None`.
- Immediately after the cross-signal `for csf in cs_factors or []:` loop (before `module_statuses = {...}`), add:

```python
        for ef in (extra_factors or []):
            fname = ef.get("factor")
            if fname:
                self.add_factor(fname, ef.get("detail", ""), factors, score)
```

`app/main.py` in `_screen_document_impl`, right after line 530 (`mrz_result["module_state"] = ...`), add:

```python
        watchlist_match = None
        mrz_data_inner = mrz_result.get("data")
        if isinstance(mrz_data_inner, dict):
            doc_number = mrz_data_inner.get("passport_number") or mrz_data_inner.get("document_number")
            if doc_number:
                watchlist_repo: Optional[WatchlistRepository] = getattr(
                    state, "watchlist_repo", None)
                if watchlist_repo is not None:
                    watchlist_match = watchlist_repo.match(doc_number)
        extra_risk_factors = None
        if watchlist_match is not None:
            extra_risk_factors = [{
                "factor": "ON_WATCHLIST",
                "detail": f"Document matched watchlist entry {watchlist_match.id} "
                          f"(severity {watchlist_match.severity}).",
            }]
```

- Change the `risk_engine.evaluate(...)` call (lines 542-549) to append `extra_factors=extra_risk_factors,` as the last arg.
- In the `response` dict (line ~554+), add after `"risk_assessment": risk,`:

```python
            "watchlist": None if watchlist_match is None else {
                "match": True,
                "watchlist_id": watchlist_match.id,
                "severity": watchlist_match.severity,
            },
```

- Ensure `from app.db.repositories import WatchlistRepository` and `Optional` are imported in `main.py`.

- [ ] **Step 4: Run to verify it passes**

Run: `C:\Python314\python -m pytest tests/test_watchlist_integration.py -v`
Expected: all PASS. Also run the risk-engine unit tests to confirm no regression: `C:\Python314\python -m pytest tests/ -k "risk" -q`.

- [ ] **Step 5: Review checkpoint** — no commit (global constraint).

---

### Task 5: Notifications data layer + real endpoint (replace Screening-derived stub)

**Files:**
- Modify: `app/db/models.py` (append `Notification`), `app/db/repositories.py` (append `NotificationRepository`), `app/main.py` (wire repo, swap `get_notifications` to DB, add `PATCH /api/v1/notifications/{id}/read`, insert high-risk notification after screening persist)
- Test: `tests/test_notifications.py`

**Interfaces:**
- Consumes: `Database`, `utcnow_naive`, `Screening` model.
- Produces:
  - `NotificationRepository(database)` with:
    - `list(*, limit: int = 10, offset: int = 0, unread_only: bool = False) -> Tuple[int, list[Notification]]`
    - `mark_read(notification_id: int) -> Optional[Notification]`
    - `create(*, type: str, message: str, screening_id: Optional[int] = None) -> Notification`
    - `create_for_screening(screening: Screening) -> Optional[Notification]` (idempotent; `None` unless high-risk)
  - `GET /api/v1/notifications` reads from the table (same `NotificationItem` shape); `PATCH /api/v1/notifications/{notification_id}/read` returns updated `NotificationItem`; response adds `"persistence": {"notification_id": ...}` on high-risk screens.

- [ ] **Step 1: Write the failing test** — `tests/test_notifications.py`:

```python
def test_notification_repository_flow(tmp_path):
    from app.db.database import build_database
    from app.db.models import Screening
    from app.db.repositories import NotificationRepository, ScreeningRepository
    db = build_database(f"sqlite:///{(tmp_path / 'n.db').as_posix()}")
    screen_repo = ScreeningRepository(db)
    notif_repo = NotificationRepository(db)

    low = screen_repo.create(
        request_id="LOW-001", processing_time_ms=10, document_type="PASSPORT",
        mrz_status="VALID", face_status="MATCH", face_similarity=0.9,
        tampering_status="CLEAN", tampering_score=0.0, liveness_status="LIVE",
        risk_score=5, risk_level="LOW_RISK", decision="CLEARED", status_color="GREEN",
        module_states={}, factor_list=[], mrz_source="form")
    assert notif_repo.create_for_screening(low) is None

    high = screen_repo.create(
        request_id="HIGH-001", processing_time_ms=20, document_type="PASSPORT",
        mrz_status="INVALID", face_status="MISMATCH", face_similarity=0.1,
        tampering_status="SUSPICIOUS", tampering_score=90.0,
        risk_score=95, risk_level="HIGH_RISK",
        decision="HIGH_RISK_REVIEW_REQUIRED", status_color="RED",
        module_states={}, factor_list=[], mrz_source="form")
    note = notif_repo.create_for_screening(high)
    assert note is not None and note.read is False
    again = notif_repo.create_for_screening(high)
    assert again is None  # idempotent

    total, rows = notif_repo.list(unread_only=True)
    assert total == 1 and rows[0].id == note.id
    marked = notif_repo.mark_read(note.id)
    assert marked is not None and marked.read is True
    assert notif_repo.mark_read(999999) is None
    total, _ = notif_repo.list(unread_only=True)
    assert total == 0
```

API test:

```python
def test_notifications_endpoint_flows(app_and_client):
    _, client = app_and_client
    token = client.post("/api/v1/auth/register", json={
        "username": "notif_o", "email": "notif@example.com",
        "full_name": "N O", "password": "Password123!",
    }).json()
    token = client.post("/api/v1/auth/login", json={
        "username": "notif_o", "password": "Password123!",
    }).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    listing = client.get("/api/v1/notifications", headers=headers)
    assert listing.status_code == 200 and listing.json() == []

    # Manufacture a high-risk screen via the repository, then read it back.
    app = app_and_client[0]
    screen_repo = app.state.screening_repo
    notif_repo = app.state.notification_repo
    from datetime import datetime
    screen_repo.create(
        request_id="API-HIGH-1", processing_time_ms=10, document_type="PASSPORT",
        mrz_status="INVALID", face_status="MISMATCH", face_similarity=0.0,
        tampering_status="SUSPICIOUS", tampering_score=90.0,
        risk_score=95, risk_level="HIGH_RISK",
        decision="HIGH_RISK_REVIEW_REQUIRED", status_color="RED",
        module_states={}, factor_list=[], mrz_source="form",
        created_at=datetime.utcnow())
    notif_repo.create_for_screening(screen_repo.get_by_request_id("API-HIGH-1"))

    items = client.get("/api/v1/notifications?unread_only=true", headers=headers).json()
    assert len(items) == 1
    nid = items[0]["id"]
    patched = client.patch(f"/api/v1/notifications/{nid}/read", headers=headers)
    assert patched.status_code == 200 and patched.json()["read"] is True
    assert client.get("/api/v1/notifications?unread_only=true",
                      headers=headers).json() == []
    assert client.patch("/api/v1/notifications/999999/read",
                        headers=headers).status_code == 404
    assert client.get("/api/v1/notifications",
                      headers={}).status_code == 401
```

- [ ] **Step 2: Run to verify it fails**

Run: `C:\Python314\python -m pytest tests/test_notifications.py -v`
Expected: FAIL (imports missing; `PATCH /api/v1/notifications/...` 405; empty-table list still returns Screening-derived rows).

- [ ] **Step 3: Implement**

`app/db/models.py` — append:

```python
class Notification(Base):
    """Persisted dashboard alert linked to a screening (read-state owned here)."""

    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    type = Column(String(40), nullable=False, default="HIGH_RISK_ALERT")
    message = Column(Text, nullable=False, default="")
    read = Column(Boolean, nullable=False, default=False)
    screening_id = Column(Integer, ForeignKey("screenings.id", ondelete="CASCADE"), nullable=True)

    __table_args__ = (
        Index("ix_notifications_created_at", created_at),
        Index("ix_notifications_read", read),
    )
```

`app/db/repositories.py` — append:

```python
class NotificationRepository:
    """Persisted notifications backed by a real table (no more Screening derivation)."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def list(self, *, limit: int = 10, offset: int = 0,
             unread_only: bool = False) -> Tuple[int, list[Notification]]:
        stmt = select(Notification).order_by(Notification.id.desc())
        if unread_only:
            stmt = stmt.where(Notification.read.is_(False))
        with self._database.session() as session:
            total = int(session.execute(
                select(func.count()).select_from(Notification)
                .where(Notification.read.is_(False) if unread_only else True)).scalar_one())
            rows = session.execute(stmt.limit(limit).offset(offset)).scalars().all()
            return total, list(rows)

    def mark_read(self, notification_id: int) -> Optional[Notification]:
        with self._database.session() as session:
            note = session.get(Notification, notification_id)
            if note is None:
                return None
            note.read = True
            session.commit()
            session.refresh(note)
            return note

    def create(self, *, type: str, message: str,
               screening_id: Optional[int] = None) -> Notification:
        with self._database.session() as session:
            note = Notification(
                created_at=utcnow_naive(), type=type, message=message,
                read=False, screening_id=screening_id,
            )
            session.add(note)
            session.commit()
            session.refresh(note)
            return note

    def create_for_screening(self, screening: Screening) -> Optional[Notification]:
        """Insert one unread HIGH_RISK_ALERT per high-risk screening (idempotent)."""
        is_high_risk = (
            screening.decision == "HIGH_RISK_REVIEW_REQUIRED"
            or screening.risk_level in {"HIGH_RISK", "CRITICAL"}
            or (screening.risk_score or 0) >= 65
            or screening.status_color == "RED"
        )
        if not is_high_risk:
            return None
        with self._database.session() as session:
            existing = session.execute(
                select(Notification.id).where(Notification.screening_id == screening.id)
            ).scalar_one_or_none()
            if existing is not None:
                return None
            note = Notification(
                created_at=utcnow_naive(), type="HIGH_RISK_ALERT",
                message=f"High risk detected on screening {screening.request_id[:8] if screening.request_id else 'UNKNOWN'}",
                read=False, screening_id=screening.id,
            )
            session.add(note)
            session.commit()
            session.refresh(note)
            return note
```

Add `Notification` to the `repositories.py` import from `app.db.models`.

`app/main.py`:
- Construct `notification_repo = NotificationRepository(database)` next to the other repos; `app.state.notification_repo = notification_repo`; import `NotificationRepository`.
- Replace the `get_notifications` endpoint body (lines 838-847) with:

```python
    @app.get("/api/v1/notifications", response_model=list[NotificationItem])
    def get_notifications(
        request: Request,
        limit: int = Query(10, ge=1, le=100),
        offset: int = Query(0, ge=0),
        unread_only: bool = Query(False),
        _current_user=Depends(get_current_user),
    ) -> list[NotificationItem]:
        repo: NotificationRepository = request.app.state.notification_repo
        _, rows = repo.list(limit=limit, offset=offset, unread_only=unread_only)
        return [NotificationItem(
            id=str(n.id), type=n.type, message=n.message,
            created_at=n.created_at.isoformat() if n.created_at else "",
            read=bool(n.read),
        ) for n in rows]

    @app.patch("/api/v1/notifications/{notification_id}/read",
               response_model=NotificationItem)
    def mark_notification_read(
        request: Request,
        notification_id: int,
        _current_user=Depends(get_current_user),
    ) -> NotificationItem:
        repo: NotificationRepository = request.app.state.notification_repo
        note = repo.mark_read(notification_id)
        if note is None:
            raise HTTPException(status_code=404, detail="Notification not found.")
        return NotificationItem(
            id=str(note.id), type=note.type, message=note.message,
            created_at=note.created_at.isoformat() if note.created_at else "",
            read=bool(note.read),
        )
```

- In `_screen_document_impl`, immediately after the existing `response["persistence"] = {"status": "stored", "screening_id": screening.id}` line, add:

```python
            notif_repo: Optional[NotificationRepository] = getattr(
                state, "notification_repo", None)
            if notif_repo is not None:
                try:
                    note = notif_repo.create_for_screening(screening)
                    if note is not None:
                        response["persistence"]["notification_id"] = note.id
                except Exception:  # noqa: BLE001 - notifications are best-effort
                    logger.error("notification_create_failed", request_id=request_id)
```

Make sure the `Notification` model import is not needed in `main.py` (only `NotificationRepository`).

- [ ] **Step 4: Run to verify it passes**

Run: `C:\Python314\python -m pytest tests/test_notifications.py -v`
Expected: all PASS. Also confirm the OLD notifications contract still type-checks: run `C:\Python314\python -m pytest tests/test_dashboard_extensions.py -q` (its notifications assertions may rely on the old derived behaviour — if a test fails, re-check the file's expectations at `tests/test_dashboard_extensions.py` and align semantics deliberately, in writing, before adjusting).

- [ ] **Step 5: Review checkpoint** — no commit (global constraint).

---

### Task 6: Ledger core (hash chain service + repository)

**Files:**
- Create: `app/services/ledger.py`
- Modify: `app/db/models.py` (append `LedgerEntry`)
- Test: `tests/test_ledger.py`

**Interfaces:**
- Consumes: `Database`, `utcnow_naive`.
- Produces:
  - `canonical_json(payload: dict) -> str`, `sha256_hex(text: str) -> str`, `compute_entry_hash(*, prev_hash, entry_type, payload, created_at) -> str`, constant `GENESIS_PREV_HASH = "0" * 64`
  - `LedgerRepository(database)` with `head()`, `seed_genesis()`, `append(*, entry_type, payload, request_id=None, created_at=None)`, `list(*, limit=50, offset=0)`, `get(entry_index)`, `verify()`.

- [ ] **Step 1: Write the failing test** — `tests/test_ledger.py`:

```python
import datetime

import pytest

from app.db.repositories import LedgerRepository  # import added in Task 6
from app.services.ledger import (
    GENESIS_PREV_HASH, canonical_json, compute_entry_hash, sha256_hex,
)
from app.db.database import build_database


def test_canonical_json_is_key_sorted():
    a = canonical_json({"b": 1, "a": {"z": 2, "y": 1}})
    b = canonical_json({"a": {"y": 1, "z": 2}, "b": 1})
    assert a == b


def test_seed_genesis_only_once(tmp_path):
    repo = LedgerRepository(build_database(f"sqlite:///{(tmp_path / 'l.db').as_posix()}"))
    g1 = repo.seed_genesis(created_at=datetime.datetime(2026, 9, 1, 0, 0, 0))
    assert g1 is not None
    assert g1.entry_index == 0
    assert g1.entry_type == "GENESIS"
    assert g1.prev_hash == GENESIS_PREV_HASH
    assert repo.seed_genesis() is None
    assert repo.head().entry_index == 0


def test_append_chains_and_verifies(tmp_path):
    repo = LedgerRepository(build_database(f"sqlite:///{(tmp_path / 'l2.db').as_posix()}"))
    repo.seed_genesis(created_at=datetime.datetime(2026, 9, 1))
    e1 = repo.append(entry_type="SCREENING", payload={"risk_score": 10},
                     request_id="A", created_at=datetime.datetime(2026, 9, 2))
    e2 = repo.append(entry_type="SCREENING", payload={"risk_score": 95},
                     request_id="B", created_at=datetime.datetime(2026, 9, 3))
    assert e2.prev_hash == e1.entry_hash
    assert repo.head().entry_index == 2
    check = repo.verify()
    assert check["valid"] is True and check["checked"] == 3 and check["broken_at"] is None


def test_verify_detects_tampered_payload(tmp_path):
    from sqlalchemy import update
    from app.db.models import LedgerEntry
    path = f"sqlite:///{(tmp_path / 'l3.db').as_posix()}"
    db = build_database(path)
    repo = LedgerRepository(db)
    repo.seed_genesis(created_at=datetime.datetime(2026, 9, 1))
    repo.append(entry_type="SCREENING", payload={"risk_score": 10}, request_id="A",
                created_at=datetime.datetime(2026, 9, 2))
    with db.session() as session:
        session.execute(update(LedgerEntry).where(LedgerEntry.entry_index == 1)
                        .values(payload={"risk_score": 999}))
        session.commit()
    check = repo.verify()
    assert check["valid"] is False and check["broken_at"] == 1


def test_list_get_offsets(tmp_path):
    repo = LedgerRepository(build_database(f"sqlite:///{(tmp_path / 'l4.db').as_posix()}"))
    repo.seed_genesis(created_at=datetime.datetime(2026, 9, 1))
    for i in range(5):
        repo.append(entry_type="X", payload={"i": i}, created_at=datetime.datetime(2026, 9, 2, i))
    total, rows = repo.list(limit=2, offset=1)
    assert total == 6
    assert [r.entry_index for r in rows] == [4, 3]
    assert repo.get(2).payload == {"i": 1}
    assert repo.get(99) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `C:\Python314\python -m pytest tests/test_ledger.py -v`
Expected: FAIL (no `app.services.ledger`, no `LedgerRepository`).

- [ ] **Step 3: Implement**

`app/db/models.py` — append:

```python
class LedgerEntry(Base):
    """One immutable hash-chain record; payload stores risk metadata only (never PII)."""

    __tablename__ = "ledger_entries"

    id = Column(Integer, primary_key=True)
    entry_index = Column(Integer, unique=True, nullable=False)
    prev_hash = Column(String(64), nullable=False)
    entry_hash = Column(String(64), nullable=False)
    payload = Column(JSON, nullable=False, default=dict)
    entry_type = Column(String(24), nullable=False, default="SCREENING")
    created_at = Column(DateTime(timezone=True), nullable=False)
    request_id = Column(String(64), nullable=True)

    __table_args__ = (
        Index("ix_ledger_entry_index", entry_index),
    )
```

`app/services/ledger.py`:

```python
"""
Append-only, privacy-preserving hash-chain ledger.

Records a SHA-256 digest of each screening's *risk metadata* (no names, no
document numbers, no raw MRZ text). The chain is deterministic: each entry
hashes its predecessor's hash plus canonical JSON of its payload, so any edit
to any past payload breaks verification.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from typing import Any, Optional, Tuple

from sqlalchemy import func, select

from app.db.database import utcnow_naive
from app.db.models import LedgerEntry

GENESIS_PREV_HASH = "0" * 64


def canonical_json(payload: dict) -> str:
    """Deterministic JSON serialization (sorted keys, compact separators)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_entry_hash(*, prev_hash: str, entry_type: str,
                       payload: dict, created_at: datetime.datetime) -> str:
    body = "|".join([prev_hash, entry_type, canonical_json(payload),
                     created_at.isoformat()])
    return sha256_hex(body)


class LedgerRepository:
    def __init__(self, database) -> None:
        self._database = database

    def head(self) -> Optional[LedgerEntry]:
        with self._database.session() as session:
            return session.execute(
                select(LedgerEntry).order_by(LedgerEntry.entry_index.desc()).limit(1)
            ).scalar_one_or_none()

    def seed_genesis(self, created_at: Optional[datetime.datetime] = None) -> Optional[LedgerEntry]:
        if self.head() is not None:
            return None
        created_at = created_at or utcnow_naive()
        entry = LedgerEntry(
            entry_index=0, prev_hash=GENESIS_PREV_HASH,
            entry_hash=compute_entry_hash(
                prev_hash=GENESIS_PREV_HASH, entry_type="GENESIS",
                payload={}, created_at=created_at),
            payload={}, entry_type="GENESIS", created_at=created_at,
        )
        with self._database.session() as session:
            session.add(entry)
            session.commit()
            session.refresh(entry)
            return entry

    def append(self, *, entry_type: str, payload: dict,
               request_id: Optional[str] = None,
               created_at: Optional[datetime.datetime] = None) -> LedgerEntry:
        created_at = created_at or utcnow_naive()
        head = self.head()
        index = (head.entry_index + 1) if head is not None else 0
        prev_hash = head.entry_hash if head is not None else GENESIS_PREV_HASH
        entry = LedgerEntry(
            entry_index=index, prev_hash=prev_hash,
            entry_hash=compute_entry_hash(
                prev_hash=prev_hash, entry_type=entry_type,
                payload=payload, created_at=created_at),
            payload=payload, entry_type=entry_type, created_at=created_at,
            request_id=request_id,
        )
        with self._database.session() as session:
            session.add(entry)
            session.commit()
            session.refresh(entry)
            return entry

    def list(self, *, limit: int = 50, offset: int = 0) -> Tuple[int, list[LedgerEntry]]:
        with self._database.session() as session:
            total = int(session.execute(
                select(func.count()).select_from(LedgerEntry)).scalar_one())
            rows = session.execute(
                select(LedgerEntry).order_by(LedgerEntry.entry_index.desc())
                .limit(limit).offset(offset)
            ).scalars().all()
            return total, list(rows)

    def get(self, entry_index: int) -> Optional[LedgerEntry]:
        with self._database.session() as session:
            return session.execute(
                select(LedgerEntry).where(LedgerEntry.entry_index == entry_index)
            ).scalar_one_or_none()

    def verify(self) -> dict[str, Any]:
        """Walk the chain; True only if every hash links and genesis is well-formed."""
        with self._database.session() as session:
            rows = session.execute(
                select(LedgerEntry).order_by(LedgerEntry.entry_index)
            ).scalars().all()
        expected_prev = GENESIS_PREV_HASH
        for entry in rows:
            if entry.entry_index == 0:
                if entry.prev_hash != GENESIS_PREV_HASH:
                    return {"valid": False, "checked": 0, "broken_at": 0}
            elif entry.prev_hash != expected_prev:
                return {"valid": False, "checked": entry.entry_index,
                        "broken_at": entry.entry_index}
            recomputed = compute_entry_hash(
                prev_hash=entry.prev_hash, entry_type=entry.entry_type,
                payload=dict(entry.payload or {}), created_at=entry.created_at)
            if recomputed != entry.entry_hash:
                return {"valid": False, "checked": entry.entry_index,
                        "broken_at": entry.entry_index}
            expected_prev = entry.entry_hash
        return {"valid": True, "checked": len(rows), "broken_at": None}
```

Add `LedgerEntry` to the `app/services/ledger.py` import (`from app.db.models import LedgerEntry`).

- [ ] **Step 4: Run to verify it passes**

Run: `C:\Python314\python -m pytest tests/test_ledger.py -v`
Expected: all PASS.

- [ ] **Step 5: Review checkpoint** — no commit (global constraint).

---

### Task 7: Ledger API + screening wiring

**Files:**
- Modify: `app/main.py` (construct + seed `LedgerRepository`, `GET /api/v1/ledger*` endpoints, append-after-persist wiring)
- Test: `tests/test_ledger_api.py`

**Interfaces:**
- Consumes: `LedgerRepository` from Task 6.
- Produces: `GET /api/v1/ledger/head`, `GET /api/v1/ledger`, `GET /api/v1/ledger/verify`; screening response `persistence` gains `ledger_index`.

- [ ] **Step 1: Write the failing test** — `tests/test_ledger_api.py`:

```python
import json

VALID_LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
UNEXPIRED_LINE2 = "L898902C36UTO7408122F3501014ZE184226B<<<<<16"


def _screen(client, headers):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (800, 500), "white").save(buf, "JPEG")
    return client.post(
        "/api/v1/screen",
        files={"document_image": ("passport.jpg", buf.getvalue(), "image/jpeg")},
        data={"mrz_line1": VALID_LINE1, "mrz_line2": UNEXPIRED_LINE2},
        headers=headers,
    )


def test_ledger_endpoints_and_privacy(app_and_client):
    _, client = app_and_client
    token = client.post("/api/v1/auth/register", json={
        "username": "ledger_o", "email": "ledger@example.com",
        "full_name": "L O", "password": "Password123!",
    }).get("username")
    token = client.post("/api/v1/auth/login", json={
        "username": "ledger_o", "password": "Password123!",
    }).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    head = client.get("/api/v1/ledger/head", headers=headers)
    assert head.status_code == 200
    assert head.json()["entry_type"] == "GENESIS"
    assert head.json()["entry_index"] == 0

    resp = _screen(client, headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["persistence"]["ledger_index"] == 1

    verify = client.get("/api/v1/ledger/verify", headers=headers)
    assert verify.json() == {"valid": True, "checked": 2, "broken_at": None}

    listing = client.get("/api/v1/ledger?limit=5&offset=0", headers=headers)
    assert listing.status_code == 200
    items = listing.json()["items"]
    assert listing.json()["total"] >= 2
    assert items[0]["entry_type"] == "SCREENING"
    # Privacy: no document number / no names anywhere in the ledger payloads.
    dumped = json.dumps(listing.json())
    for secret in ("L898902C", "ANNA MARIA", "maria"):
        assert secret not in dumped

    assert client.get("/api/v1/ledger", headers={}).status_code == 401
```

- [ ] **Step 2: Run to verify it fails**

Run: `C:\Python314\python -m pytest tests/test_ledger_api.py -v`
Expected: FAIL (`GET /api/v1/ledger/head` 404; `ledger_index` absent).

- [ ] **Step 3: Implement**

`app/main.py`:
- Import: `from app.services.ledger import LedgerRepository` (or `from app.db.repositories import ...`? No — `LedgerRepository` lives in `app/services/ledger.py`, not repositories).
- Construction block (Task 3 location): add `ledger_repo = LedgerRepository(database)`; `app.state.ledger_repo = ledger_repo`; right after the schema-ensure block call `ledger_repo.seed_genesis()`.
- Endpoints, placed near the other dashboard endpoints:

```python
    # ---- Ledger (hash chain) ----------------------------------------------
    @app.get("/api/v1/ledger/head")
    def ledger_head(request: Request,
                    _current_user=Depends(get_current_user)) -> Optional[dict]:
        repo: LedgerRepository = request.app.state.ledger_repo
        head = repo.head()
        if head is None:
            return None
        return {
            "entry_index": head.entry_index,
            "entry_hash": head.entry_hash,
            "prev_hash": head.prev_hash,
            "entry_type": head.entry_type,
            "created_at": head.created_at.isoformat() if head.created_at else "",
            "payload": head.payload,
        }

    @app.get("/api/v1/ledger")
    def ledger_list(request: Request,
                    limit: int = Query(50, ge=1, le=500),
                    offset: int = Query(0, ge=0),
                    _current_user=Depends(get_current_user)) -> dict:
        repo: LedgerRepository = request.app.state.ledger_repo
        total, rows = repo.list(limit=limit, offset=offset)
        items = [{
            "entry_index": e.entry_index,
            "entry_hash": e.entry_hash,
            "prev_hash": e.prev_hash,
            "entry_type": e.entry_type,
            "created_at": e.created_at.isoformat() if e.created_at else "",
            "payload": e.payload,
            "request_id": e.request_id,
        } for e in rows]
        return {"items": items, "total": total}

    @app.get("/api/v1/ledger/verify")
    def ledger_verify(request: Request,
                      _current_user=Depends(get_current_user)) -> dict:
        repo: LedgerRepository = request.app.state.ledger_repo
        return repo.verify()
```

- In `_screen_document_impl`, after the `response["persistence"] = {"status": "stored", "screening_id": screening.id}` line (before or after the notification block from Task 5), add:

```python
            ledger_repo: Optional[LedgerRepository] = getattr(state, "ledger_repo", None)
            if ledger_repo is not None:
                try:
                    ledger_payload = {
                        "request_id": request_id,
                        "screening_id": screening.id,
                        "document_type": str(response["document"].get("type", "UNKNOWN")),
                        "mrz_status": str(mrz_result.get("status", "NOT_DETECTED")),
                        "face_status": str(face_result.get("status", "NOT_AVAILABLE")),
                        "tampering_status": str(tamper_result.get("status", "CLEAN")),
                        "liveness_status": str(liveness_result.get("liveness_status", "NOT_CHECKED")),
                        "risk_score": int(risk["score"]),
                        "risk_level": str(risk["level"]),
                        "decision": str(risk["decision"]),
                        "status_color": str(risk["status"]),
                        "watchlist_hit": bool(watchlist_match is not None),
                    }
                    ledger_entry = ledger_repo.append(
                        entry_type="SCREENING", payload=ledger_payload,
                        request_id=request_id)
                    response["persistence"]["ledger_index"] = ledger_entry.entry_index
                except Exception:  # noqa: BLE001 - ledger is best-effort
                    logger.error("ledger_append_failed", request_id=request_id)
```

- Import `Optional` if not already in `main.py`.

- [ ] **Step 4: Run to verify it passes**

Run: `C:\Python314\python -m pytest tests/test_ledger_api.py -v`
Expected: all PASS.

- [ ] **Step 5: Review checkpoint** — no commit (global constraint).

---

### Task 8: CSV export

**Files:**
- Modify: `app/db/repositories.py` (add `ScreeningRepository.export_rows`), `app/main.py` (`GET /api/v1/report/export.csv`)
- Test: `tests/test_csv_export.py`

**Interfaces:**
- Consumes: `ScreeningRepository`, `_normalize_utc` (existing private helper).
- Produces: `ScreeningRepository.export_rows(*, date_from=None, date_to=None, limit=None) -> list[dict]`; `GET /api/v1/report/export.csv` -> UTF-8 BOM CSV attachment.

- [ ] **Step 1: Write the failing test** — `tests/test_csv_export.py`:

```python
import datetime

VALID_LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
UNEXPIRED_LINE2 = "L898902C36UTO7408122F3501014ZE184226B<<<<<16"


def _screen(client, headers):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (800, 500), "white").save(buf, "JPEG")
    return client.post(
        "/api/v1/screen",
        files={"document_image": ("passport.jpg", buf.getvalue(), "image/jpeg")},
        data={"mrz_line1": VALID_LINE1, "mrz_line2": UNEXPIRED_LINE2},
        headers=headers,
    )


def test_export_csv_bom_headers_and_rows(app_and_client):
    _, client = app_and_client
    token = client.post("/api/v1/auth/register", json={
        "username": "csv_o", "email": "csv@example.com",
        "full_name": "C O", "password": "Password123!",
    }).get("username")
    token = client.post("/api/v1/auth/login", json={
        "username": "csv_o", "password": "Password123!",
    }).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    resp = _screen(client, headers)
    assert resp.status_code == 200
    request_id = resp.json()["request_id"] or resp.json().get("request_id")

    out = client.get("/api/v1/report/export.csv", headers=headers)
    assert out.status_code == 200
    assert out.headers["content-type"] == "text/csv; charset=utf-8"
    assert out.headers["content-disposition"].startswith("attachment; filename=")
    raw = out.content
    assert raw[:3] == b"\xef\xbb\xbf"  # UTF-8 BOM
    text = raw.decode("utf-8-sig")
    lines = text.splitlines()
    header = lines[0]
    assert "request_id" in header and "risk_score" in header and "decision" in header
    assert request_id[:8] in text  # request_id is the correlation token (allowed)


def test_export_never_contains_doc_numbers(app_and_client):
    _, client = app_and_client
    token = client.post("/api/v1/auth/register", json={
        "username": "csv_o2", "email": "csv2@example.com",
        "full_name": "C O2", "password": "Password123!",
    }).get("username")
    token = client.post("/api/v1/auth/login", json={
        "username": "csv_o2", "password": "Password123!",
    }).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert _screen(client, headers).status_code == 200
    text = client.get("/api/v1/report/export.csv",
                      headers=headers).content.decode("utf-8-sig")
    assert "L898902C" not in text
    assert "ANNA MARIA" not in text


def test_export_requires_auth(app_and_client):
    _, client = app_and_client
    assert client.get("/api/v1/report/export.csv").status_code == 401


def test_export_respects_date_filter(app_and_client):
    # Should still return rows without dates, and 0 rows for a far-future window.
    _, client = app_and_client
    token = client.post("/api/v1/auth/register", json={
        "username": "csv_o3", "email": "csv3@example.com",
        "full_name": "C O3", "password": "Password123!",
    }).get("username")
    token = client.post("/api/v1/auth/login", json={
        "username": "csv_o3", "password": "Password123!",
    }).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert _screen(client, headers).status_code == 200
    ok = client.get(
        "/api/v1/report/export.csv",
        params={"date_from": "2020-01-01", "date_to": "2099-12-31"},
        headers=headers)
    assert ok.status_code == 200 and len(ok.content.decode("utf-8-sig").splitlines()) >= 2
    empty = client.get(
        "/api/v1/report/export.csv",
        params={"date_from": "2999-01-01", "date_to": "2999-12-31"},
        headers=headers)
    assert len(empty.content.decode("utf-8-sig").splitlines()) == 1  # header only
```

- [ ] **Step 2: Run to verify it fails**

Run: `C:\Python314\python -m pytest tests/test_csv_export.py -v`
Expected: FAIL (`404 Not Found` on `/api/v1/report/export.csv`).

- [ ] **Step 3: Implement**

`app/db/repositories.py` — append to `ScreeningRepository`:

```python
    def export_rows(self, *, date_from: Optional[object] = None,
                    date_to: Optional[object] = None,
                    limit: Optional[int] = None) -> list[dict]:
        filters = []
        if date_from:
            filters.append(Screening.created_at >= _normalize_utc(date_from))
        if date_to:
            filters.append(Screening.created_at <= _normalize_utc(date_to))
        with self._database.session() as session:
            stmt = (select(Screening).where(*filters)
                    .order_by(Screening.created_at, Screening.id))
            if limit is not None:
                stmt = stmt.limit(limit)
            return [
                {
                    "screening_id": s.id,
                    "request_id": s.request_id,
                    "created_at": s.created_at.isoformat() if s.created_at else "",
                    "processing_time_ms": s.processing_time_ms,
                    "document_type": s.document_type,
                    "mrz_status": s.mrz_status,
                    "face_status": s.face_status,
                    "face_similarity": s.face_similarity,
                    "tampering_status": s.tampering_status,
                    "tampering_score": s.tampering_score,
                    "liveness_status": s.liveness_status,
                    "liveness_score": s.liveness_score,
                    "risk_score": s.risk_score,
                    "risk_level": s.risk_level,
                    "decision": s.decision,
                    "status_color": s.status_color,
                    "mrz_source": s.mrz_source,
                    "notes": s.notes or "",
                }
                for s in session.execute(stmt).scalars().all()
            ]
```

`app/main.py` — new endpoint near `/api/v1/report/summary`:

```python
    CSV_EXPORT_COLUMNS = [
        "screening_id", "request_id", "created_at", "processing_time_ms",
        "document_type", "mrz_status", "face_status", "face_similarity",
        "tampering_status", "tampering_score", "liveness_status",
        "liveness_score", "risk_score", "risk_level", "decision",
        "status_color", "mrz_source", "notes",
    ]

    @app.get("/api/v1/report/export.csv")
    def export_csv(
        request: Request,
        date_from: Optional[datetime.date] = None,
        date_to: Optional[datetime.date] = None,
        _current_user=Depends(get_current_user),
    ) -> Response:
        repo: ScreeningRepository = request.app.state.screening_repo
        rows = repo.export_rows(date_from=date_from, date_to=date_to)
        import csv
        import io as _io
        buffer = _io.StringIO()
        buffer.write("\ufeff")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(CSV_EXPORT_COLUMNS)
        for row in rows:
            writer.writerow([row.get(col, "") for col in CSV_EXPORT_COLUMNS])
        filename = (f"document_screening_export_"
                    f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv")
        return Response(
            content=buffer.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
```

Check `datetime` and `Response` are imported in `main.py` (they are: `from datetime import datetime` and `from fastapi import Response` should exist; if not, add them).

- [ ] **Step 4: Run to verify it passes**

Run: `C:\Python314\python -m pytest tests/test_csv_export.py -v`
Expected: all PASS.

- [ ] **Step 5: Review checkpoint** — no commit (global constraint).

---

### Task 9: PDF document upload (pymupdf render-first-page)

**Files:**
- Create: `app/services/pdf.py`
- Modify: `requirements.txt` (add `pymupdf>=1.24`), `app/main.py` (`_screen_document_impl` PDF injection + `converted_from_pdf` flag)
- Test: `tests/test_pdf_upload.py`

**Interfaces:**
- Consumes: `settings.max_pdf_bytes`, `settings.pdf_render_max_dimension` (Task 1); `validation_limits`.
- Produces:
  - `app.services.pdf.is_pdf_content_type(content_type=None, filename=None) -> bool`
  - `app.services.pdf.render_first_page(pdf_bytes, *, max_dimension=4096, jpeg_quality=80) -> Optional[bytes]`
  - Screening response `document.converted_from_pdf` (`bool`).

- [ ] **Step 1: Install dependency + write failing test**

First install the new dependency used by this task:
`C:\Python314\python -m pip install "pymupdf>=1.24"`

Then `tests/test_pdf_upload.py`:

```python
import io

from app.services.pdf import is_pdf_content_type, render_first_page

VALID_LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
UNEXPIRED_LINE2 = "L898902C36UTO7408122F3501014ZE184226B<<<<<16"


def _pdf_bytes():
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 portrait
    page.insert_text((72, 120), "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<<<<<<<<",
                      fontname="helv", fontsize=8)
    return doc.tobytes()


def test_is_pdf_detection():
    assert is_pdf_content_type("application/pdf", "scan.pdf") is True
    assert is_pdf_content_type("", "doc.pdf") is True
    assert is_pdf_content_type("image/jpeg", "photo.jpg") is False


def test_render_first_page_returns_jpeg():
    data = render_first_page(_pdf_bytes())
    assert data is not None and data[:2] == b"\xff\xd8"


def test_render_rejects_invalid_bytes():
    assert render_first_page(b"this is not a pdf") is None
    assert render_first_page(b"") is None


def test_screen_pdf_reports_converted(app_and_client):
    _, client = app_and_client
    token = client.post("/api/v1/auth/register", json={
        "username": "pdf_o", "email": "pdf@example.com",
        "full_name": "P O", "password": "Password123!",
    }).get("username")
    token = client.post("/api/v1/auth/login", json={
        "username": "pdf_o", "password": "Password123!",
    }).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    resp = client.post(
        "/api/v1/screen",
        files={"document_image": ("scan.pdf", _pdf_bytes(), "application/pdf")},
        data={"mrz_line1": VALID_LINE1, "mrz_line2": UNEXPIRED_LINE2},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["document"]["converted_from_pdf"] is True


def test_screen_invalid_pdf_rejected(app_and_client):
    _, client = app_and_client
    token = client.post("/api/v1/auth/register", json={
        "username": "pdf_o2", "email": "pdf2@example.com",
        "full_name": "P O2", "password": "Password123!",
    }).get("username")
    token = client.post("/api/v1/auth/login", json={
        "username": "pdf_o2", "password": "Password123!",
    }).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    resp = client.post(
        "/api/v1/screen",
        files={"document_image": ("fake.pdf", b"not really a pdf", "application/pdf")},
        data={"mrz_line1": VALID_LINE1, "mrz_line2": UNEXPIRED_LINE2},
        headers=headers,
    )
    assert resp.status_code == 422


def test_screen_image_has_no_pdf_flag(app_and_client):
    import io as _io
    from PIL import Image
    _, client = app_and_client
    token = client.post("/api/v1/auth/register", json={
        "username": "pdf_o3", "email": "pdf3@example.com",
        "full_name": "P O3", "password": "Password123!",
    }).get("username")
    token = client.post("/api/v1/auth/login", json={
        "username": "pdf_o3", "password": "Password123!",
    }).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    buf = _io.BytesIO()
    Image.new("RGB", (800, 500), "white").save(buf, "JPEG")
    resp = client.post(
        "/api/v1/screen",
        files={"document_image": ("id.jpg", buf.getvalue(), "image/jpeg")},
        data={"mrz_line1": VALID_LINE1, "mrz_line2": UNEXPIRED_LINE2},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["document"].get("converted_from_pdf") is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `C:\Python314\python -m pytest tests/test_pdf_upload.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'app.services.pdf'`).

- [ ] **Step 3: Implement**

`app/services/pdf.py`:

```python
"""PDF upload support: detect PDF uploads and render the first page to a JPEG
so the existing image pipeline can screen scanned PDF documents."""

from __future__ import annotations

from typing import Optional


def is_pdf_content_type(content_type: Optional[str] = None,
                        filename: Optional[str] = None) -> bool:
    ct = (content_type or "").lower()
    name = (filename or "").lower()
    return ct == "application/pdf" or name.endswith(".pdf") or name.endswith(".PDF")


def render_first_page(pdf_bytes: bytes, *, max_dimension: int = 4096,
                      jpeg_quality: int = 80) -> Optional[bytes]:
    """Render page 1 as a JPEG at bounded resolution; None on any failure."""
    if not pdf_bytes:
        return None
    try:
        import fitz  # pymupdf
    except Exception:  # noqa: BLE001
        return None
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            if document.page_count < 1:
                return None
            page = document.load_page(0)
            rect = page.rect
            max_side = max(rect.width, rect.height) or 1.0
            zoom = min(max_dimension / max_side, 2.0)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            return pixmap.tobytes("jpeg", jpg_quality=jpeg_quality)
        finally:
            document.close()
    except Exception:  # noqa: BLE001 - invalid/encrypted PDFs must never 500
        return None
```

`requirements.txt` — add a line `pymupdf>=1.24` (in the existing dependency list, matching formatting).

`app/main.py` — in `_screen_document_impl`, replace the upload-read block (lines 433-452) with:

```python
        from app.services import pdf as pdf_util
        is_pdf = pdf_util.is_pdf_content_type(
            document_image.content_type, document_image.filename)
        try:
            read_limit = (settings.max_pdf_bytes if is_pdf
                          else validation_limits.max_bytes) + 1
            raw_doc_bytes = await document_image.read(read_limit)
            if is_pdf and len(raw_doc_bytes) > settings.max_pdf_bytes:
                raise HTTPException(
                    status_code=413,
                    detail="PDF exceeds the maximum allowed size.")
            if is_pdf:
                doc_bytes = pdf_util.render_first_page(
                    raw_doc_bytes,
                    max_dimension=settings.pdf_render_max_dimension)
                if doc_bytes is None:
                    raise HTTPException(
                        status_code=422,
                        detail="The uploaded PDF could not be rendered; "
                               "upload a clean single-page scan or an image.")
                effective_content_type = "image/jpeg"
                effective_filename = "document_page_1.jpg"
            else:
                doc_bytes = raw_doc_bytes
                effective_content_type = document_image.content_type
                effective_filename = document_image.filename
            validation_limits.validate(
                doc_bytes, effective_content_type, "Document image",
                effective_filename)
            live_bytes: Optional[bytes] = None
            if live_photo:
                live_bytes = await live_photo.read(validation_limits.max_bytes + 1)
                validation_limits.validate(live_bytes, live_photo.content_type,
                                           "Live photo", live_photo.filename)
        finally:
            await document_image.close()
            if live_photo:
                await live_photo.close()
```

- In the `response` `document` dict (line ~558-562), add the flag:
  `"converted_from_pdf": bool(is_pdf),`
- **Fix the `ScreenResponse` schema** so the new key serializes: check `app/models/schemas.py` `ScreenResponse`; if its `document` field is a strict `BaseModel` without `converted_from_pdf`, add `converted_from_pdf: bool = False` to that nested model. (Read the schema first; a plain `dict`-typed document field needs no change.)
- Note: `doc_bytes`/`effective_*`/`is_pdf` must remain in scope for the rest of the function; the `try/finally` only closes the upload handles.

- [ ] **Step 4: Run to verify it passes**

Run: `C:\Python314\python -m pytest tests/test_pdf_upload.py -v`
Expected: all PASS.

- [ ] **Step 5: Review checkpoint** — no commit (global constraint).

---

### Task 10: ICAO 9303 TD2 visa parser + routing

**Files:**
- Modify: `app/services/mrz.py` (add `_td2_lines_valid`, `parse_td2_visa`, `VisaTD2Parser`, aliases, router registration, resolve message), `app/main.py` (`allowed_document_types` + error copy)
- Test: `tests/test_visa_td2.py`

**Interfaces:**
- Consumes: existing `calculate_icao_checksum`, `verify_mrz_field`, `valid_date`, `mrz_year_full`, `_valid_mrz_chars`, `DocumentParseResult`, `BaseDocumentParser`, `_clean_mrz_line`.
- Produces: `parse_td2_visa(line1, line2, year_pivot=50) -> dict` and `VisaTD2Parser` (`name="visa"`, `document_type="VISA"`, `format="TD2"`), registered under aliases `"td2"`/`"visa"`; `document_type` values `td2`/`visa` accepted by `POST /api/v1/screen`.

- [ ] **Step 1: Write the failing test** — `tests/test_visa_td2.py`:

```python
from app.services.mrz import (
    DocumentParserRouter, VisaTD2Parser, calculate_icao_checksum, parse_td2_visa,
)

# Deterministic, fully-compliant TD2 vector built exactly per ICAO 9303 Part 4.
# Line 2: doc(9)+chk | nationality(3) | dob(6)+chk | sex(1) | expiry(6)+chk |
#         optional(7) | composite-chk(1). Composite covers
#         line2[0:10] + line2[13:20] + line2[21:28] + line2[28:35].
LINE1 = ("VBIND" + "TEST<<PERSON" + "<" * (36 - len("VBIND") - len("TEST<<PERSON")))
_DOC = "123456789"
_DOB = "900101"
_EXP = "301231"
_OPT = "ASDFGHJ"
_BODY = (_DOC + str(calculate_icao_checksum(_DOC)) + "IND"
         + _DOB + str(calculate_icao_checksum(_DOB)) + "M"
         + _EXP + str(calculate_icao_checksum(_EXP)) + _OPT)
_COMPOSITE = _BODY[0:10] + _BODY[13:20] + _BODY[21:28] + _BODY[28:35]
LINE2 = _BODY + str(calculate_icao_checksum(_COMPOSITE))

assert len(LINE1) == 36 and len(LINE2) == 36


def test_parse_valid_td2_visa():
    parsed = parse_td2_visa(LINE1, LINE2)
    assert parsed["status"] == "VALID"
    assert parsed["doc_type"] == "VB"
    assert parsed["issuing_country"] == "IND"
    assert parsed["document_number"] == "123456789"
    assert parsed["full_name"] == "PERSON TEST"
    assert parsed["nationality"] == "IND"
    assert parsed["date_of_birth"] == "900101"
    assert parsed["gender"] == "M"
    assert parsed["expiry_date"] == "301231"
    assert parsed["optional_data"] == "ASDFGHJ"
    assert all(parsed["checks"].values())
    assert parsed["is_expired"] is False


def test_parse_detects_corrupted_digit():
    corrupt = list(LINE2)
    corrupt[0] = "9" if corrupt[0] != "9" else "8"
    parsed = parse_td2_visa(LINE1, "".join(corrupt))
    assert parsed["status"] == "INVALID"
    assert parsed["checks"]["document_number_valid"] is False


def test_parse_malformed_length():
    assert parse_td2_visa("TOOSHORT", LINE2)["status"] == "MALFORMED"


def test_expired_visa_detected():
    from app.services.mrz import calculate_icao_checksum
    doc = "123456789"; chk = calculate_icao_checksum(doc)
    dob = "900101"; dob_chk = calculate_icao_checksum(dob)
    exp = "200101"; exp_chk = calculate_icao_checksum(exp)  # past expiry
    opt = "ASDFGHJ"
    composite = (doc + str(chk) + "IND" + dob + str(dob_chk) + "M"
                 + exp + str(exp_chk) + opt)
    comp_chk = calculate_icao_checksum(composite)
    line2 = composite + str(comp_chk)
    parsed = parse_td2_visa(LINE1, line2)
    assert parsed["status"] == "VALID"
    assert parsed["is_expired"] is True


def test_parser_from_lines_detected():
    parser = VisaTD2Parser()
    result = parser.parse(None, line1=LINE1, line2=LINE2)
    assert result.detected is True
    assert result.status == "VALID"
    assert result.document_type == "VISA"
    assert result.format == "TD2"
    assert result.data["document_number"] == "123456789"


def test_router_aliases():
    router = DocumentParserRouter()
    assert router.resolve("visa") == "visa"
    assert router.resolve("td2") == "visa"
    assert router.resolve("passport") == "passport"
```

Note on `test_expired_visa_detected`: build a line2 whose PAYLOAD is unchanged but the **expiry check digit is recalculated correctly** for the past date, so the document is still checksum-valid but marked expired:

```python
    from app.services.mrz import calculate_icao_checksum
    doc = "123456789"; chk = calculate_icao_checksum(doc)
    dob = "900101"; dob_chk = calculate_icao_checksum(dob)
    exp = "200101"; exp_chk = calculate_icao_checksum(exp)
    opt = "ASDFGHJ"
    composite = (doc + str(chk) + "IND" + dob + str(dob_chk) + "M"
                 + exp + str(exp_chk) + opt)
    comp_chk = calculate_icao_checksum(composite)
    line2 = composite + str(comp_chk)
    parsed = parse_td2_visa(LINE1, line2)
    assert parsed["status"] == "VALID"
    assert parsed["is_expired"] is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `C:\Python314\python -m pytest tests/test_visa_td2.py -v`
Expected: FAIL (`ImportError: cannot import name 'parse_td2_visa'`).

- [ ] **Step 3: Implement**

`app/services/mrz.py`:

- In `PARSER_ALIASES` add: `"td2": "visa", "visa": "visa",`.
- Add functions after `parse_td1_national_id`:

```python
def _td2_lines_valid(line1: str, line2: str) -> bool:
    return (
        len(line1) == 36
        and len(line2) == 36
        and _valid_mrz_chars(line1)
        and _valid_mrz_chars(line2)
        and line2[10:13].isalpha()
        and line2[13:19].isdigit()
        and line2[21:27].isdigit()
        and line2[20] in {"M", "F", "<"}
        and valid_date(line2[13:19])
        and valid_date(line2[21:27])
    )


def parse_td2_visa(line1: str, line2: str, year_pivot: int = 50) -> dict[str, Any]:
    """Parse and validate ICAO 9303 TD2 (2x36) visa lines.

    Line 1: 0-1 doc type, 2-4 issuing country, 5-35 name field (31 chars).
    Line 2: 0-8 document number, 9 check; 10-12 nationality; 13-18 DOB, 19 check;
    20 sex; 21-26 expiry, 27 check; 28-34 optional data, 35 composite check.
    """
    if len(line1) != 36 or len(line2) != 36:
        return {"status": "MALFORMED",
                "error": "TD2 lines must each contain exactly 36 characters."}
    if not _td2_lines_valid(line1, line2):
        return {"status": "MALFORMED",
                "error": "TD2 structure or characters are invalid."}

    doc_type = line1[0:2].replace("<", "")
    issuing_country = line1[2:5].replace("<", "")
    name_field = line1[5:36]
    name_parts = name_field.split("<<")
    surname = name_parts[0].replace("<", " ").strip()
    given_names = name_parts[1].replace("<", " ").strip() if len(name_parts) > 1 else ""

    document_number = line2[0:9].replace("<", "")
    dob_raw = line2[13:19]
    gender = line2[20].replace("<", "X")
    expiry_raw = line2[21:27]
    optional_data = line2[28:35].replace("<", " ").strip()

    valid_number = verify_mrz_field(line2[0:9], line2[9])
    valid_dob = verify_mrz_field(dob_raw, line2[19])
    valid_expiry = verify_mrz_field(expiry_raw, line2[27])
    composite_data = line2[0:10] + line2[13:20] + line2[21:28] + line2[28:35]
    valid_composite = verify_mrz_field(composite_data, line2[35])

    birth_date_valid = valid_date(dob_raw, year_pivot=year_pivot)
    expiry_date_valid = valid_date(expiry_raw, year_pivot=year_pivot)
    try:
        exp_date = datetime.date(
            mrz_year_full(int(expiry_raw[0:2]), year_pivot=year_pivot),
            int(expiry_raw[2:4]),
            int(expiry_raw[4:6]),
        )
        is_expired = exp_date < datetime.date.today()
    except ValueError:
        is_expired = False

    checks = {
        "document_number_valid": valid_number,
        "dob_valid": valid_dob and birth_date_valid,
        "expiry_valid": valid_expiry and expiry_date_valid,
        "composite_valid": valid_composite,
    }
    status = "VALID" if all(checks.values()) else "INVALID"

    return {
        "status": status,
        "doc_type": doc_type,
        "issuing_country": issuing_country,
        "full_name": f"{given_names} {surname}".strip(),
        "document_number": document_number,
        "nationality": line2[10:13].replace("<", ""),
        "date_of_birth": dob_raw,
        "gender": gender,
        "expiry_date": expiry_raw,
        "optional_data": optional_data,
        "is_expired": is_expired,
        "checks": checks,
    }
```

- Add the `VisaTD2Parser` class after `NationalIDTD1Parser` (mirroring it, but 2×36 and a light OCR path that only accepts explicit-length 36 candidates). `parse` prefers `_from_lines` when both lines are supplied; otherwise collects 36-char candidates via `_clean_mrz_line` and returns on the first valid pair (no tesseract required for tests; OCR path is best-effort only):

```python
class VisaTD2Parser(BaseDocumentParser):
    """ICAO 9303 TD2 (2x36) visa parser. Explicit type only (td2/visa)."""

    name = "visa"
    document_type = "VISA"
    format = "TD2"

    def can_parse(self, image_bgr: np.ndarray) -> bool:
        if image_bgr is None or image_bgr.size == 0:
            return False
        height, width = image_bgr.shape[:2]
        if height <= 0:
            return False
        return (width / height) >= PASSPORT_MIN_RATIO

    def parse(self, image, settings=None, line1=None, line2=None, line3=None,
              **kwargs):
        if settings is None:
            settings = Settings.from_env()
        if line1 is not None and line2 is not None:
            return self._from_lines(line1, line2, settings)
        image_bytes: bytes = image if isinstance(image, bytes) else _encode_bgr(image)
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("L")
            width, height = image.size
            crop = image.crop((0, int(height * 0.55), width, height))
            crop = crop.resize((width * 2, max(1, crop.height * 2)))
            try:
                ocr_text = pytesseract.image_to_string(
                    ImageEnhance.Contrast(crop).enhance(2.0),
                    config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
            finally:
                image.close()
                crop.close()
            lines = [_clean_mrz_line(l) for l in ocr_text.splitlines()
                     if len(_clean_mrz_line(l)) == 36]
            unique = list(dict.fromkeys(lines))
            for i, first in enumerate(unique):
                for j, second in enumerate(unique):
                    if i == j:
                        continue
                    if not _td2_lines_valid(first, second):
                        continue
                    parsed = parse_td2_visa(first, second,
                                            year_pivot=settings.mrz_year_pivot)
                    if parsed.get("status") == "VALID":
                        return self._from_lines(first, second, settings)
        except Exception:
            raw = {"detected": False, "source": "ocr", "status": "OCR_FAILED",
                   "confidence": 0.0, "reason": "MRZ OCR unavailable; secondary inspection required."}
            return self._result(raw)
        raw = {"detected": False, "source": "ocr", "status": "NOT_DETECTED",
               "confidence": 0.0,
               "reason": "No valid TD2 (2x36) visa MRZ could be extracted."}
        return self._result(raw)

    def _from_lines(self, line1: str, line2: str, settings: Settings) -> DocumentParseResult:
        parsed = parse_td2_visa(line1, line2, year_pivot=settings.mrz_year_pivot)
        checks = parsed.get("checks")
        checks_valid = bool(isinstance(checks, dict) and all(checks.values()))
        status = str(parsed.get("status", "MALFORMED"))
        raw = {
            "detected": bool(checks_valid),
            "source": "form",
            "status": status,
            "confidence": 1.0 if checks_valid else 0.5,
            "line1": line1,
            "line2": line2,
            "data": parsed,
            "validation": {
                "structure_valid": bool(status not in {"MALFORMED", "INVALID"}),
                "checksums_valid": checks_valid,
                "dates_valid": bool(isinstance(checks, dict)
                                    and checks.get("dob_valid")
                                    and checks.get("expiry_valid")),
            },
        }
        return self._result(raw)

    def _result(self, raw: dict[str, Any]) -> DocumentParseResult:
        result = dict(raw)
        result["format"] = self.format
        result["document_type"] = self.document_type
        return DocumentParseResult(
            detected=bool(result.get("detected", False)),
            status=str(result.get("status", "NOT_DETECTED")),
            document_type=self.document_type,
            format=self.format,
            confidence=float(result.get("confidence", 0.0)),
            data=dict(result.get("data", {})),
            raw=result,
            error=None if result.get("detected") else str(result.get("reason", "")),
        )
```

- In `DocumentParserRouter._ensure_parser`, add before the `unknown` branch:

```python
            elif key == "visa":
                from app.services.mrz import VisaTD2Parser
                self._parsers["visa"] = VisaTD2Parser()
```

(Since `VisaTD2Parser` is defined in the same module, reference it directly: `self._parsers["visa"] = VisaTD2Parser()`.)

- Update `resolve()` error text to: `"document_type must be one of: auto, td3/passport, td2/visa, td1/national_id, aadhaar, pan."`

`app/main.py`:
- Line 424: `allowed_document_types = {"auto", "td3", "passport", "td2", "visa", "td1", "national_id", "aadhaar", "pan"}`
- Line 429-431 error copy: `"document_type must be one of: auto, td3/passport, td2/visa, td1/national_id, aadhaar, pan."`

- [ ] **Step 4: Run to verify it passes**

Run: `C:\Python314\python -m pytest tests/test_visa_td2.py -v`
Expected: all PASS.

- [ ] **Step 5: Review checkpoint** — no commit (global constraint).

---

### Task 11: Full-suite verification

**Files:** none.

- [ ] **Step 1: Run the full test suite**

Run: `C:\Python314\python -m pytest -q`
Expected: 248 existing + new tests all PASS (a green total well above 248).

- [ ] **Step 2: Run ruff on the changed surface**

Run: `C:\Python314\python -m ruff check app tests`
Expected: no new violations (pre-existing `main.py` F401/F841 may remain — do NOT fix).

- [ ] **Step 3: Demonstrate no commit**

Run: `git status` and confirm new/modified files are present but nothing is staged/committed. Report to the user: new features complete, zero commits made, and ask whether/how they want to commit + deploy (user decides; never push on your own).

- [ ] **Step 4: Optional ad-hoc verification** — start the API locally and smoke the four GET endpoints:

```
C:\Python314\python -m uvicorn app.main:app --port 8000
```

Then check `GET /api/v1/ledger/head` (Bearer token) → genesis row; `GET /api/v1/ledger/verify` → `{"valid": true, ...}`.

---

## Self-Review

**1. Spec coverage** — every approved feature has a task: ledger hash-chain (Tasks 6-7), real watchlists CRUD + match + risk factor (Tasks 2-4), real notifications (Task 5), CSV export (Task 8), PDF upload (Task 9), Visa TD2 parser (Task 10), config/extras (Task 1). PDF-liveness-model-plumbing and email alerts were explicitly NOT in scope — correctly absent. Watchlist match is risk-factor + flag only (Task 4), no auto-HOLD — correct per approval.

**2. Placeholder scan** — every step has concrete code or an exact anchor; the one "write remaining cases" line in Task 2 names the four specific cases and the exact setup pattern to copy, not an unimplemented procedure.

**3. Type/signature consistency** — `WatchlistRepository.match/create/update/delete/list/search/seed_defaults` signatures are identical between Task 2 (definition) and Task 3/4 (call sites). `LedgerRepository.head/seed_genesis/append/list/get/verify` match Task 6 definitions with Task 7 call sites. `NotificationRepository.list/mark_read/create/create_for_screening` match Task 5 definitions and endpoint call sites. `extra_factors` defaults to `None` so untouched callers are unaffected. `parse_td2_visa` return keys (`document_number`, `full_name`, `checks`, `is_expired`) line up with the TD1/TD3 parsers so the Task 4 doc-number extraction (`passport_number` or `document_number`) covers visas automatically. `render_first_page`/`is_pdf_content_type` signatures match main.py injection points.