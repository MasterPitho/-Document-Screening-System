# Sentinel screening frontend

This is the local dashboard frontend for the Document Screening Engine.

## Local testing

Do **not** open `frontend.html` with `file://`.

The dashboard uses `server.py` as a local static-file server and reverse proxy so browser requests to `/api/*` are forwarded to the configured backend.

### Windows

Double-click:

```text
start-dashboard.bat
```

It starts the local proxy on port `3000` and opens:

```text
http://localhost:3000/frontend.html
```

### Manual start

From this directory:

```powershell
python server.py
```

Then open:

```text
http://localhost:3000/frontend.html
```

The current proxy target is configured in `server.py`. The dashboard requires an authenticated officer session before document screening because `POST /api/v1/screen` requires a Bearer token.

### Upload requirements

The screening API accepts document images in **JPG, PNG, or WebP** format, up to 10 MB by default. PDF files are not supported by the screening endpoint.

The frontend sends:

- `document_image` — required document image
- `live_photo` — optional face image
- `document_type` — selected parser type

The frontend stores only the authentication session locally; document images are sent to the screening API for in-memory processing and are not intended to be persisted by the engine.
