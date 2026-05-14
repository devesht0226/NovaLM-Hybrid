# NovaLM-Hybrid

Mini decoder-only Transformer language model with optional TF-IDF retrieval augmentation.

**Full project overview (purpose, stack, layout, workflows):** see [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md).

## Problem Statement

Pure small language models often produce generic or weakly grounded outputs on factual prompts.  
`NovaLM-Hybrid` addresses this by combining:
- **Pure LM generation** for baseline fluency
- **Retrieval-augmented generation** for stronger grounding using top-k relevant chunks

## Project status

This repository is a **complete end-to-end implementation**: preprocessing, tokenizer training, LM training, retrieval indexing, FastAPI backend, SQLite history, web UI, automated compare/scoring scripts, Docker, and tests. It is **not** a partial prototype or single-notebook demo.

It is built as a **portfolio- or coursework-grade deliverable** with clear documented limits (small transformer, TF-IDF retrieval). **Metrics in the README and final report** come from the current `best.pt`; re-run `scripts/run_eval.py` and compare scripts after retraining to refresh them. It is **not** meant as turnkey enterprise production (managed scaling, authentication, dense semantic retrieval, etc. are out of scope unless you extend it).

**Continuous integration** runs unit tests that do not require trained weights (see **Testing**). Full API tests run on your machine after you produce `artifacts/checkpoints/best.pt`, `artifacts/tokenizers/tokenizer.model`, and `data/index/chunks.json`.

### Remaining gaps (manual or optional)

| Item | What to do |
|------|------------|
| **Manual rubric scores** | Optional if you use `proxy_scores.py`. Otherwise fill `artifacts/reports/scores.csv`, then run `python scripts/score_report.py`. |
| Demo **screenshots** | Capture per **Demo Evidence Checklist** if your course or portfolio requires them. |
| **CI vs full tests** | GitHub Actions runs `pytest -m "not integration"` only; full `pytest` needs local artifacts. |
| **Quality ceiling** | Small LM + TF-IDF; better answers need more data, training, or a denser retriever (see **Limitations and Next Steps**). |

## Why This Is Better Than Old Project

- Built a complete product pipeline, not only model experimentation.
- Added dual generation modes: pure LM and hybrid retrieval-augmented LM.
- Added production-style API validation, latency tracking, and error handling.
- Added persistent SQLite history with filtering, pagination, delete, and export.
- Added integrated frontend for live generation and context inspection.
- Added automated comparison and scoring workflow for repeatable evaluation.

## Architecture

Raw text -> preprocess -> train tokenizer -> tokenize -> train LM -> serve:
- `/generate`: pure language model generation
- `/hybrid_generate`: retrieve top-k context chunks + generate

## Architecture Flow (Simple)

1. Ingest and preprocess corpus text.
2. Train SentencePiece tokenizer and build token IDs.
3. Train mini decoder-only Transformer and save best checkpoint.
4. Build retrieval index from processed text chunks.
5. Serve FastAPI endpoints for pure/hybrid generation.
6. Persist requests/results in SQLite history.
7. Use frontend for prompting, comparison, and history operations.

## Setup

```bash
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -e .
```

## Data Preparation

Pipeline scripts resolve relative paths against the **project root** (the folder that contains `configs/`), so you can run them from any working directory as long as the package is installed (`pip install -e .`).

1. Put `.txt` files inside `data/raw/`
2. Run:

```bash
$env:PYTHONPATH='.'
python scripts/run_preprocess.py
python scripts/run_tokenizer.py
python scripts/run_build_token_ids.py
```

`run_preprocess.py` now skips automatically when the configured processed file already exists. Use `python scripts/run_preprocess.py --force` when you want to rebuild.

## Train

Hyperparameters default from `configs/model.yaml` and `configs/train.yaml` (learning rate, warmup, cosine schedule, early stopping, gradient accumulation). Override via CLI flags such as `--epochs`, `--batch_size`, `--lr`, `--sp_model` (for vocab size when omitting `--vocab_size`).

```bash
$env:PYTHONPATH='.'
python scripts/run_train.py
```

## Evaluate

```bash
$env:PYTHONPATH='.'
python scripts/run_eval.py
```

## Testing

Without trained artifacts (matches CI):

```bash
python -m pytest -m "not integration"
```

`tests/conftest.py` skips collecting `tests/test_api.py` in that mode so the API module (which loads `best.pt` at import time) is never imported. To force collection of API tests in CI with a checkpoint, set `RUN_INTEGRATION_API=1` and provide weights.

Full suite including HTTP/API checks (requires checkpoint, SentencePiece model, and `chunks.json` on disk):

```bash
python -m pytest
```

With the API already running, end-to-end HTTP checks (no extra dependencies beyond the app):

```bash
$env:PYTHONPATH='.'
python scripts/smoke_release.py --base-url http://127.0.0.1:8080
```

## Run API

From the repo root, using the bundled helpers (sets `PYTHONPATH` and uses `.venv` when present):

```powershell
.\scripts\run_api.ps1
.\scripts\run_api.ps1 --reload
```

```bash
chmod +x scripts/run_api.sh   # once, on Unix
./scripts/run_api.sh
./scripts/run_api.sh --reload
```

Development (auto-reload), manual:

```bash
$env:PYTHONPATH='.'
uvicorn src.serving.api:app --host 0.0.0.0 --port 8080 --reload
```

Production-style (no reload; tune workers as needed):

```bash
$env:PYTHONPATH='.'
uvicorn src.serving.api:app --host 0.0.0.0 --port 8080
```

## Docker

Build and run with mounted checkpoints, tokenizer, retrieval index, and DB (paths match `.env.example` defaults):

```bash
docker compose build
docker compose up
```

Optional: copy `.env.example` to `.env` to override paths or set **`NOVALM_HOST_PORT`** (maps host port to container `8080`; default `8080`).

Then open `http://127.0.0.1:8080` (or `http://127.0.0.1:<NOVALM_HOST_PORT>` if set). Ensure `artifacts/checkpoints/best.pt`, `artifacts/tokenizers/tokenizer.model`, and `data/index/chunks.json` exist on the host before starting. The image uses a **healthcheck** on `/health`; first startup can take up to a couple of minutes while the model loads.

## Inference examples

```bash
curl -X GET "http://localhost:8080/health"
```

```bash
curl -X POST "http://localhost:8080/generate" -H "Content-Type: application/json" -d "{\"prompt\":\"Explain attention\"}"
```

```bash
curl -X POST "http://localhost:8080/hybrid_generate" -H "Content-Type: application/json" -d "{\"prompt\":\"Explain attention\",\"retrieval_top_k\":3}"
```

Debug retrieval only (no generation):

```bash
curl "http://localhost:8080/retrieve_debug?q=attention&k=5"
```

## History and Export API

```bash
curl "http://localhost:8080/history?limit=20&offset=0"
curl "http://localhost:8080/history?mode=hybrid&q=attention"
curl "http://localhost:8080/history/export?fmt=json"
curl "http://localhost:8080/history/export?fmt=csv"
```

## Compare + Report Workflow

Generate pure-vs-hybrid outputs, `artifacts/reports/compare_rows.json`, retrieval samples, and refreshed latencies (merges existing **Metrics** / **Score Averages** / **Best Cases** blocks from `final_report.md` when present):

```bash
$env:PYTHONPATH='.'
python scripts/run_compare.py
```

Optional: **heuristic** scores (lexical overlap / repetition / context overlap — not a human rubric) into `scores.csv` and the prompt table:

```bash
$env:PYTHONPATH='.'
python scripts/proxy_scores.py
python scripts/score_report.py
```

Create an empty scoring sheet (manual rubric) instead:

```bash
$env:PYTHONPATH='.'
python scripts/score_report.py --init-template
```

After manually scoring `artifacts/reports/scores.csv`, inject averages into the report:

```bash
$env:PYTHONPATH='.'
python scripts/score_report.py
```

With the API running, capture JSON evidence (health, generate, hybrid, history, retrieve) under `artifacts/reports/evidence/`:

```powershell
.\scripts\collect_evidence.ps1
```

## Evaluation Summary

Numbers below are from **`artifacts/checkpoints/best.pt`** and the current splits (`data/processed/token_ids.json`, `data/splits/test.txt`). Regenerate after retraining:

```bash
$env:PYTHONPATH='.'
python scripts/run_eval.py
python scripts/run_compare.py
```

- **Validation loss** (recomputed on `val_ids`, same checkpoint): **6.5750**
- **Validation perplexity**: **716.94**
- **Test loss** (`test.txt` via tokenizer): **6.5645**
- **Test perplexity**: **709.44**
- **Avg latency** pure (`/generate`, last `run_compare.py` run, `--max-new-tokens 32`): **348.45 ms**
- **Avg latency** hybrid (`/hybrid_generate`, same run): **1358.57 ms** (retrieval + longer decode; varies by CPU/GPU and token count)
- **Winner (latency, that run):** Pure generation path was faster; hybrid pays retrieval cost.
- **Winner (quality, heuristic proxy scores):** **Pure** overall **2.93** vs hybrid **2.70** (see `artifacts/reports/final_report.md` **Score Averages**). Replace with your own `scores.csv` rubric if required.

## Compared To Previous Project

| Area | Previous Project | NovaLM-Hybrid |
|---|---|---|
| Scope | Model-focused experiment | End-to-end product pipeline |
| Generation Mode | Pure LM only | Pure + Hybrid retrieval mode |
| API Validation | Basic/minimal | Strict schema validation |
| Persistence | None or ad hoc | SQLite history with filters/pagination/export |
| Frontend | Limited or absent | Integrated interactive UI |
| Evaluation | Manual-only | Automated compare + scoring scripts |
| Deployment Readiness | Low | Moderate (API + env config + export + docs) |

## Notes

- Artifacts expected by API: `artifacts/checkpoints/best.pt` and `artifacts/tokenizers/tokenizer.model`
- On very small datasets, tokenizer trains with a reduced effective vocabulary automatically.
- Copy `.env.example` to `.env` for local path overrides; variables are applied when `src.utils.config` loads (via `python-dotenv`, without overriding variables already set in your shell).
- Hybrid retrieval is configured in **`configs/retrieval.yaml`** (TF‑IDF options, optional `min_context_score` to skip weak matches, optional custom `hybrid_instruction` template with `{excerpts}` and `{prompt}`).

## Submission Bundle Checklist

Include:
- `src/`
- `scripts/`
- `configs/`
- `tests/`
- `README.md`
- `PROJECT_GUIDE.md` (single-file deep overview for reviewers)
- `pyproject.toml`
- `Dockerfile`, `docker-compose.yml` (optional container run)
- `.github/workflows/ci.yml` (optional automated tests on push/PR)
- `scripts/proxy_scores.py`, `scripts/collect_evidence.ps1` (optional reporting / evidence)
- `.gitignore`
- `.env.example`
- `artifacts/reports/final_report.md`

Exclude:
- `.venv/`
- temporary logs/cache
- machine-specific editor/system files

## Demo Evidence Checklist

Capture screenshots for:
- training completion and best checkpoint saved
- API health success (`/health`)
- `/generate` success
- `/hybrid_generate` success with retrieved contexts
- frontend history filter/search/delete/export actions

## 2-Minute Viva Pitch

"NovaLM-Hybrid improves on my previous language model project by moving from a model-only prototype to a complete end-to-end system.  
I implemented the full pipeline: preprocessing, tokenizer training, Transformer LM training, retrieval indexing, and dual-mode generation (pure + hybrid).  
The hybrid path injects retrieved context before generation, improving practical relevance and grounding for knowledge-heavy prompts.  
Beyond modeling, I added a production-style FastAPI backend with strict validation, latency tracking, global error handling, and SQLite history persistence with filters, pagination, and export.  
I also built an integrated frontend for live prompting, context inspection, and history operations, plus automated evaluation scripts for repeatable comparison and scoring.  
So compared to my old project, this is more modular, reproducible, testable, and demo-ready as a usable product." 

## Limitations and Next Steps

- Quality scales with corpus size and training time; tiny demos stay noisy by design.
- Retrieval uses TF-IDF only.
- Decoding supports top-k, optional nucleus top-p, and repetition penalty via `configs/infer.yaml` or request fields.
- Next: denser retriever, richer evaluation, and deployment hardening beyond the provided Docker baseline.
