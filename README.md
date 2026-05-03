# Conference AI Assistant

A sales-prep tool: snap a photo of a customer's business card, and the system
extracts the contact, researches their company on the web, and produces a
PDF brief with likely pain points and three opening questions.

This is **v1** — the card-scan + company-research slice. Live conversation
hints and post-meeting summaries are planned for v2 and v3.

## Architecture at a glance

```
[ iOS / web ]  ──photo──▶  [ FastAPI ]  ──▶  Claude Vision (extract fields)
                                       ──▶  Claude + web_search (research)
                                       ──▶  reportlab (PDF)
                                       ◀──  status + result + PDF
```

- `backend/` — FastAPI service, SQLite for state, local disk for uploads/reports.
- `web/` — single-file HTML tester. Served by FastAPI at `/app/`.

Key files:

- `backend/app/extraction.py` — Claude Vision call with forced tool use.
- `backend/app/research.py` — Claude + web_search server tool, structured brief.
- `backend/app/pipeline.py` — orchestration; runs in a FastAPI background task.
- `backend/app/pdf_report.py` — reportlab rendering.
- `backend/app/main.py` — HTTP endpoints.

## Run it locally (5 minutes)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and paste your ANTHROPIC_API_KEY.

uvicorn app.main:app --reload --port 8000
```

Open the tester at <http://localhost:8000/app/>. Upload (or snap, on a
phone) a business card; you'll see fields populate, a research brief
arrive, and a PDF download appear.

## API surface

| Method | Path                       | Purpose                                  |
|--------|----------------------------|------------------------------------------|
| POST   | `/cards`                   | Upload a card photo. Returns `{id}`.     |
| GET    | `/cards/{id}`              | Status + extracted fields + research.    |
| GET    | `/cards/{id}/report.pdf`   | Download the rendered PDF brief.         |
| GET    | `/cards/{id}/photo`        | Original uploaded photo.                 |
| POST   | `/cards/{id}/email`        | Stub — returns the email that *would* be sent. Wire SES/SendGrid here. |
| GET    | `/cards`                   | List recent cards (debug).               |

Auto-generated API docs live at `/docs`.

## Configuration (`backend/.env`)

| Var                | Default                                      | Notes |
|--------------------|----------------------------------------------|-------|
| `ANTHROPIC_API_KEY`| (required)                                   |       |
| `ANTHROPIC_MODEL`  | `claude-sonnet-4-5`                          | Override to try Haiku for cheaper extraction or Opus for tougher research. |
| `COMPANY_CONTEXT`  | d-volt voltage / power conditioning blurb    | Steers the research agent. Tune as you learn what salespeople want. |
| `DATA_DIR`         | `./data`                                     | Photos, PDFs, SQLite live here. |
| `DB_PATH`          | `./data/app.db`                              |       |

## Deploy to Render (Phase 1 of iOS rollout)

The repo has a `Dockerfile` at the root and a `render.yaml` that declares a
single web service with a 1 GB persistent disk. Steps:

1. **Push the repo to GitHub** (Render reads `render.yaml` from the
   default branch).

2. **Create the service.** In the Render dashboard, click **New →
   Blueprint**, connect this repo, pick the branch, and Apply. Render
   provisions a Docker web service, mounts a 1 GB disk at `/data`, and
   starts the build.

3. **Set the secrets.** Open the service → **Environment** tab and paste:
   - `ANTHROPIC_API_KEY` — your Anthropic key
   - `OPENAI_API_KEY` — your OpenAI key (used for Whisper)
   - `COMPANY_CONTEXT` — optional, overrides the d-volt blurb

4. **Wait for the first build.** ~2-4 minutes. Render shows live logs;
   the service is healthy once `GET /health` returns 200.

5. **Hit the URL.** Render gives you `https://conference-ai-XXXX.onrender.com`.
   Open `/app/` in your phone's browser to confirm the web tester loads
   over HTTPS. This URL is what the iOS Capacitor build will hit in
   Phase 2.

### Plans
The blueprint specifies `plan: starter` (~$7/mo). That's the cheapest tier
that gives you **always-on + persistent disk**, both required so the
SQLite DB and uploaded files survive restarts. Free tier spins down after
15 minutes of inactivity and wipes disk state, so don't use it once any
real data matters.

### Verifying the build locally before pushing
```bash
docker build -t conference-ai .
docker run --rm -p 8000:8000 \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  conference-ai
# then open http://localhost:8000/health and http://localhost:8000/app/
```

## What's deliberately **not** here yet

- Auth — anyone with the URL can call the API. Fine for local prototyping; add JWT/OAuth before exposing it.
- Email sending — `/cards/{id}/email` is a stub. SES/SendGrid wiring is one function.
- Streaming audio + live hints — that's slice v2.
- Post-meeting summary report — slice v3.
- iOS app — coming next once you sign off on the AI quality.

## Slice 2 (live hints) and 3 (post-meeting summary) — sketch

- Slice 2 will add a WebSocket endpoint `/meetings/{id}/stream`, a Deepgram
  or AssemblyAI streaming adapter, and a Claude hint engine that runs every
  ~30 seconds against a rolling transcript window.
- Slice 3 will add `/meetings/{id}/report`, generating a structured summary
  PDF the same way slice 1 generates the pre-meeting brief.

Both will reuse the storage, PDF, and Claude client patterns established here.
