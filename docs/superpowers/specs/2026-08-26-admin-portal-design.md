# Admin Portal Design
**Date:** 2026-08-26  
**Project:** GSM HR Chatbot  
**Author:** Melany Morales-Garibay

---

## Overview

Add a password-protected admin portal to the existing GSM HR Chatbot at `/admin`. The portal gives the HR team (3 people), IT, and the president the ability to view chat history, review analytics, and manage HR documents — including uploading new ones and triggering a chatbot update — without needing command-line access.

---

## Architecture

The admin panel is a protected route inside the existing React frontend, not a separate app. `main.jsx` checks `window.location.pathname` — if it starts with `/admin`, it renders the admin shell instead of the employee chatbot. All admin backend logic lives in a new file `backend/app/admin.py`, registered as a router in `main.py`.

No new Docker containers. No new infrastructure. The admin panel is invisible to employees — there are no links to `/admin` from the chatbot UI.

The nginx config must serve `index.html` for `/admin` (standard SPA fallback rule). This will be verified and added if missing.

---

## Authentication

- Single shared password stored in the server's `.env` as `ADMIN_PASSWORD`
- On login, the backend checks the submitted password against `ADMIN_PASSWORD`
- If correct, returns a **session token** — a random 48-character string generated at backend startup and held in memory
- The frontend stores the token in `localStorage` under key `gsm_admin_token`
- Every admin API request includes the token in an `Authorization: Bearer <token>` header
- A 401 response clears the stored token and returns the user to the login screen
- The token resets on every backend restart; HR logs in again with the same permanent password
- To revoke all access immediately: change `ADMIN_PASSWORD` in `.env` and restart the backend

No user accounts, no roles, no password reset flow.

---

## Backend API

All endpoints except `/login` require a valid token in the `Authorization` header.

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/admin/login` | Verify password, return session token |
| GET | `/api/admin/logs` | Return chat history CSV as JSON array |
| GET | `/api/admin/analytics` | Return topic counts, daily usage, and summary totals |
| GET | `/api/admin/documents` | List files in the documents folder |
| POST | `/api/admin/upload` | Accept a file upload (PDF, DOCX, TXT), save to documents folder |
| DELETE | `/api/admin/documents/{filename}` | Delete a document file |
| POST | `/api/admin/ingest` | Trigger re-ingest as a background task, return immediately |
| GET | `/api/admin/ingest/status` | Return whether ingest is currently running, succeeded, or failed |

The ingest endpoint uses FastAPI `BackgroundTasks` so the HTTP request returns immediately. The frontend polls `/api/admin/ingest/status` every 3 seconds until status is `done` or `error`.

---

## Analytics

Analytics are computed server-side when `/api/admin/analytics` is called by reading and processing the usage CSV.

**Summary totals returned:**
- Total questions
- Unique session count
- Answered count and percentage
- Redirected count and percentage

**Daily usage:** question count grouped by date, for the last 30 days.

**Topic detection:** keyword matching on the question text. Each question is assigned one topic — the first category whose keywords match. If none match, it falls into "Other."

| Topic | Keywords |
|-------|----------|
| PTO / Vacation | pto, vacation, time off, days off, accrual, accrued |
| Benefits / Insurance | benefit, insurance, health, dental, vision, medical, coverage |
| FMLA / Leave | fmla, leave, maternity, paternity, family leave, medical leave |
| Bereavement | bereavement, funeral, death, passing |
| Holidays | holiday, christmas, thanksgiving, labor day, memorial day, new year |
| Pay / Payroll | pay, payroll, salary, wage, overtime, direct deposit |
| Conduct / Policy | conduct, policy, disciplinary, harassment, code of conduct |
| Other | (catch-all) |

---

## Frontend Structure

```
frontend/src/
  main.jsx                  — route to AdminApp or App based on pathname
  admin/
    AdminApp.jsx            — shell: auth gate, sidebar, top bar, tab switching
    Login.jsx               — password form
    Analytics.jsx           — stat cards, daily bar chart, topic bars
    LogViewer.jsx           — paginated chat history table
    Documents.jsx           — file list, upload zone, Update Chatbot button
    adminApi.js             — all admin fetch calls with auth header
```

Token is checked on `AdminApp` mount. If absent, Login is shown. On successful login, token is saved and the dashboard renders.

---

## Document Management Flow

**Upload:**
1. HR drops a file onto the upload zone or clicks to browse
2. File is validated client-side (type: PDF/DOCX/TXT, size: ≤ 20 MB)
3. File uploads to `POST /api/admin/upload`
4. Document list refreshes — new file appears
5. Chatbot is not yet updated

**Update Chatbot:**
1. HR clicks "Update Chatbot"
2. Button changes to "Updating…" with a spinner
3. Frontend polls `/api/admin/ingest/status` every 3 seconds
4. On success: button resets, last-updated timestamp refreshes
5. On failure: error message appears with a retry option

**Remove document:**
1. HR clicks "Remove" next to a file
2. Confirmation dialog: *"Remove [filename]? This cannot be undone."*
3. On confirm: file is deleted, list refreshes
4. HR must click "Update Chatbot" for the removal to take effect in the chatbot

**Guardrails:**
- Accepted file types: PDF, DOCX, TXT
- Max file size: 20 MB
- Cannot delete the last remaining document

---

## Security Constraints

These requirements were enforced during implementation and apply to any future changes:

- **Empty-password guard:** `/api/admin/login` returns 503 if `ADMIN_PASSWORD` is not set in the environment. This prevents open access on misconfigured deployments.
- **Chunked upload:** `POST /api/admin/upload` reads the file in 64 KB chunks and aborts mid-stream if the size exceeds 20 MB. The full body is never buffered into memory before the size check.
- **Ingest race prevention:** `POST /api/admin/ingest` acquires a `threading.Lock` before checking and setting the `"running"` state. This prevents two concurrent requests from both passing the 409 guard and starting overlapping ingest runs.
- **Consistent 401 handling:** All admin API calls in `adminApi.js` throw the string `'unauthorized'` (not a generic error) on a 401 response, which triggers an automatic logout and return to the login screen.
- **Non-ok response handling:** `handleAuth()` also throws on any non-2xx response (not only 401), preventing error payloads from being silently treated as valid data.
- **Token comparison:** Token and password comparisons use `secrets.compare_digest` to avoid timing attacks.
- **Path traversal prevention:** Delete endpoint resolves the target path and verifies it is within `DOCUMENTS_DIR` before deleting.

---

## Files Changed

**Backend:**
- `backend/app/admin.py` — new file, all admin endpoints and ingest logic
- `backend/app/config.py` — `admin_password` field added to `Settings`
- `backend/app/ingest.py` — `run_ingest()` accepts optional `qdrant_host` parameter
- `backend/app/main.py` — register admin router
- `backend/requirements.txt` — `python-multipart>=0.0.5` added (required for `UploadFile`)
- `backend/tests/test_admin.py` — 15 backend tests

**Frontend:**
- `frontend/src/main.jsx` — pathname-based routing
- `frontend/src/admin/` — new directory: `adminApi.js`, `admin.css`, `Login.jsx`, `Analytics.jsx`, `LogViewer.jsx`, `Documents.jsx`, `AdminApp.jsx`

**Infrastructure:**
- `.env.example` — `ADMIN_PASSWORD=` entry added
- `.env` — add `ADMIN_PASSWORD` on server only, never committed
