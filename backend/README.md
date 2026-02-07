# ReelClaw Backend (local dev)

This is a FastAPI API + a Batch worker wrapper around the reel replication pipeline (`backend/reelclaw_pipeline/`).

## Requirements (machine running the backend)

- Python 3
- `ffmpeg` on PATH
- `yt-dlp` on PATH (used to download reference videos by URL)
- `OPENROUTER_API_KEY` in your environment (or a `./openrouter` file at repo root)

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
# API-only deps:
# pip install -r backend/requirements.txt
#
# Full pipeline deps (recommended for local E2E editing; includes Pillow/numpy/sklearn/joblib):
pip install -r backend/requirements.worker.txt
```

## Run

```bash
cd backend
REELCLAW_DEV_AUTH=1 uvicorn reelclaw_backend.main:app --host 0.0.0.0 --port 8000
```

Notes:

- iOS devices cannot reach `localhost`. Use your Mac’s LAN IP in the app’s Settings, e.g. `http://192.168.1.10:8000`.
- The app’s Debug build allows HTTP (ATS is relaxed). For production, you’ll want HTTPS.
- Instagram URLs often require authentication (cookies). For reliable tests, use **Reference → Upload** in the iOS app.

## Instagram / YouTube reference links (AWS)

In AWS, the Batch worker uses `yt-dlp` to download reference URLs. Many Instagram (and some YouTube) URLs require a logged-in session.

The worker is already wired to read a Netscape-format `cookies.txt` from AWS Secrets Manager:

- `reelclaw-prod/ytdlp_cookies` (prod)
- `reelclaw-staging/ytdlp_cookies` (staging)

Notes:

- Cookies are **best-effort** by default. If the secret is missing/empty, the worker will try to download without cookies and only fail if the source requires login.
- To fail fast when cookies are missing/misconfigured, set `REELCLAW_YTDLP_COOKIES_REQUIRED=1` in the worker environment.
- If Instagram blocks cloud IPs in your region/account, you can route `yt-dlp` through a proxy:
  - `REELCLAW_YTDLP_PROXY` (explicit proxy URL) or `REELCLAW_YTDLP_PROXY_SECRET_ID` (Secrets Manager id holding proxy URL)
  - Or set `OXYLABS_PROXY_USERNAME` / `OXYLABS_PROXY_PASSWORD` (+ host/port/protocol) to reuse Oxylabs-style config
  - To fail fast when proxy is required for Instagram, set `REELCLAW_YTDLP_PROXY_REQUIRED=1`

To set/update it, export a `cookies.txt` from a browser where you’re logged in, then run:

```bash
set -a; source .env.production; set +a
python3 backend/scripts/set_ytdlp_cookies_secret.py --env prod --cookies-file /path/to/cookies.txt
```

This is a **sensitive secret**. Do not commit it, and rotate it when it expires.

## API flow (current)

The iOS client uses a 2-step job flow:

1) `POST /v1/jobs` creates a job and returns upload URLs (either presigned S3 URLs in AWS mode, or local API PUT URLs in dev mode).
2) The client uploads the reference (optional) + clips via **PUT** to the returned `upload_url`s.
3) `POST /v1/jobs/{job_id}/start` submits the job (AWS Batch) or runs locally.

## E2E (prod/staging)

`backend/scripts/e2e_aws_flow.py` mimics the iOS → AWS flow using a short-lived JWT minted from the `jwt_secret` in AWS Secrets Manager:

```bash
python3 backend/scripts/e2e_aws_flow.py \
  --env prod \
  --region us-east-1 \
  --clips-dir /Users/work/Downloads/clipss \
  --reference-upload /Users/work/Downloads/clipss/Video-694.mp4
```

## Useful env vars

- `REELCLAW_DATA_DIR` (default: `backend/data`) – where jobs/uploads/outputs are stored.
- `REELCLAW_DIRECTOR_MODEL` – overrides the model passed to the pipeline (default: `google/gemini-3-pro-preview`).
- `REELCLAW_REFERENCE_ANALYSIS_MAX_SECONDS`
  - unset: backend uses full reel length (no cap)
  - `40`: cap analysis clip to 40 seconds
  - `0`: no cap (full reel)

### Local dev auth

- `REELCLAW_DEV_AUTH=1` enables local dev Apple Sign-In bypass (token is decoded unverified).
- In production (AWS), set `REELCLAW_DEV_AUTH=0`, `REELCLAW_APPLE_AUDIENCE=<bundle id>`, and provide `REELCLAW_JWT_SECRET`.
