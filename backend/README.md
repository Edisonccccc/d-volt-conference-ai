# Backend — FastAPI + Claude

## First run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000/app/> for the web tester, or <http://localhost:8000/docs>
for the FastAPI docs UI.

## Smoke test with curl

```bash
# Upload a card photo
curl -F "photo=@/path/to/card.jpg" http://localhost:8000/cards

# Poll status (replace ID)
curl http://localhost:8000/cards/<id> | jq

# Once status == "ready"
open http://localhost:8000/cards/<id>/report.pdf
```

## File map

```
app/
├── main.py          FastAPI endpoints + CORS + static mount for web tester
├── pipeline.py      Background task: extract -> research -> PDF
├── extraction.py    Claude Vision -> ExtractedCard (forced tool use)
├── research.py      Claude + web_search server tool -> CompanyResearch
├── pdf_report.py    reportlab rendering
├── storage.py       SQLite + filesystem
└── models.py        Pydantic types
```

## Notes for the next engineer

- The pipeline runs synchronously in a FastAPI `BackgroundTasks` worker,
  which means it shares the uvicorn process and is fine for prototyping.
  Move to a real queue (Celery/RQ/SQS) before scaling.
- `research.py` uses Claude's server-side `web_search` tool. If your
  Anthropic account doesn't have access, the call will fail with a 400 and
  the brief will fall back to a stub. You can swap in your own search
  provider by replacing `WEB_SEARCH_TOOL` and parsing tool_use blocks
  yourself.
- The PDF layout assumes letter paper. Switch `pagesizes.LETTER` to `A4` if
  you're shipping outside North America.
- Photos are stored on local disk under `data/uploads/`. For production
  move to S3 and stream to Claude rather than reading into memory.
