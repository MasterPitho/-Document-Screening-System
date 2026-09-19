# Sentinel screening frontend

Single-page dashboard for the Document Screening Engine. It talks to the API
through a small local proxy (`server.py`) so you can open it over HTTP instead
of `file://` — that also sidesteps CORS.

## Running it

Do not open `frontend.html` straight from disk. Start the proxy first:

```powershell
python server.py
```

then open `http://localhost:3000/frontend.html`.

On Windows you can also double-click `start-dashboard.bat` — it starts the
proxy on port `3000` and opens the page for you.

## Pointing at a backend

`server.py` forwards `/api/*` to whatever `BACKEND` points at. Default is the
deployed engine. For local development, override it:

```powershell
$env:BACKEND = "http://127.0.0.1:8000"
python server.py
```

`PORT` works the same way if you want a different port. Tools like curl against
the backend directly also work — the API endpoints don't know or care that the
page came through a proxy.

One more override: append `?api=...` to the page URL (e.g.
`http://localhost:3000/frontend.html?api=http://127.0.0.1:8000`) to pin the
backend from the browser. Opening the HTML via `file://` falls back to the
deployed engine URL.

## What the dashboard does

- Screen a document image (JPG/PNG/WebP up to 10 MB, or PDF up to 20 MB —
  the first page is rendered before screening). Optional live photo for face
  match. Document types: Passport (TD3), Visa (TD2), National ID (TD1),
  Aadhaar, PAN.
- Show the verdict and its risk factors broken down, plus history of past
  screenings.
- Manage watchlists — add, change severity, delete. Screenings against a
  watchlist entry get flagged with a higher risk score.
- Unread notifications you can dismiss.
- A tamper-evident ledger of screening records, with a verify button so you
  can check a recent entry hasn't been altered.
- Export the screening report as CSV.

The page needs a logged-in officer session before screening — `POST
/api/v1/screen` requires a Bearer token.

## Privacy note

Only the login session stays in the browser. Document images go straight to
the API and are processed in memory; nothing is stored client-side.