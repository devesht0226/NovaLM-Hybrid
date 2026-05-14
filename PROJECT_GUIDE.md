# NovaLM-Hybrid — Project guide (single reference)

This file is the **one-stop overview** for anyone who needs to understand what was built, **why**, the **technical stack**, how pieces connect, and how to run or extend the system. The `README.md` stays the quick start; this guide goes deeper.

---

## Table of contents

1. [Purpose and motivation](#1-purpose-and-motivation)  
2. [What the system does](#2-what-the-system-does)  
3. [Technical stack (complete)](#3-technical-stack-complete)  
4. [Repository layout](#4-repository-layout)  
5. [End-to-end data flow](#5-end-to-end-data-flow)  
6. [API surface](#6-api-surface)  
7. [Configuration and environment](#7-configuration-and-environment)  
8. [Running locally](#8-running-locally)  
9. [Docker](#9-docker)  
10. [Evaluation, reports, and evidence](#10-evaluation-reports-and-evidence)  
11. [Testing and CI](#11-testing-and-ci)  
12. [Typical failure modes and limits](#12-typical-failure-modes-and-limits)  
13. [If you build something similar](#13-if-you-build-something-similar)  
14. [Checklist before submission or demo](#14-checklist-before-submission-or-demo)

---

## 1. Purpose and motivation

**Problem:** Small language models used alone often produce **generic** or **weakly grounded** text on factual or domain-heavy prompts.

**Approach:** **NovaLM-Hybrid** combines:

- **Pure LM:** a small **decoder-only Transformer** trained on your corpus.  
- **Hybrid LM:** **retrieve** top‑k text chunks with **TF‑IDF**, then **generate** with that context so outputs can lean on **your** indexed material.

**Why this repo exists (vs. “only a notebook”):** To show a **full product pipeline**—not just training curves: preprocessing, tokenizer, training, retrieval index, **serving**, **persistence**, **UI**, **evaluation scripts**, **Docker**, and **automated tests / CI**.

**What it is not:** A replacement for large frontier chat models, or a turnkey enterprise hosted product (auth, multi-tenant scaling, observability stacks, etc. are out of scope unless you add them).

---

## 2. What the system does

| Area | Capability |
|------|------------|
| **Data** | Ingest raw `.txt`, clean, dedupe, split train/val/test, build overlapping **chunks** for retrieval, write `chunks.json`. |
| **Tokenizer** | Train **SentencePiece** on your text; build token ID files for training. |
| **Model** | Train a **mini Transformer LM** (decoder-only); save checkpoints (`best.pt`). |
| **Retrieval** | **TF‑IDF** over chunks; used in hybrid generation path. |
| **Serving** | **FastAPI** app: pure and hybrid generation, health, history, export, retrieval debug. |
| **UI** | Static **HTML/JS** in `src/serving/web`, mounted at `/static` and served at `/`. |
| **History** | **SQLite** for stored generations, filters, pagination, delete, CSV/JSON export. |
| **Ops** | **Docker** / **Compose**, optional `.env`, healthcheck. |
| **Quality** | **pytest** (unit + optional integration), **GitHub Actions** CI, `smoke_release.py`, optional `collect_evidence.ps1`. |

---

## 3. Technical stack (complete)

### Language and packaging

- **Python** `>= 3.10`  
- **setuptools** editable install: `pip install -e .`  
- Package lives under **`src/`** (see `pyproject.toml` `package-dir`).

### Dependencies (from `pyproject.toml`)

| Library | Role |
|---------|------|
| **torch** | Model implementation, training, inference, checkpoints, CPU/CUDA. |
| **sentencepiece** | Train/load subword tokenizer (`tokenizer.model`). |
| **numpy** | Numerics alongside PyTorch where needed. |
| **scikit-learn** | TF‑IDF retrieval (`TfidfVectorizer` + scoring). |
| **fastapi** | HTTP API, routing, dependency injection patterns. |
| **uvicorn[standard]** | ASGI server (production-style serve). |
| **pydantic** | Request/response validation (v2). |
| **pyyaml** | Load `configs/*.yaml`. |
| **httpx** | HTTP client (tests / tooling). |
| **python-dotenv** | Load `.env` from project root when `src.utils.config` is imported. |
| **pytest** | Automated tests. |

### Infrastructure and tooling

- **Docker** (`Dockerfile`, `docker-compose.yml`): slim Python image, bind mounts for artifacts and index.  
- **GitHub Actions** (`.github/workflows/ci.yml`): CI runs `pytest -m "not integration"` (no checkpoint in repo).  
- **PowerShell / bash** helper scripts: `scripts/run_api.ps1`, `scripts/run_api.sh`, `scripts/collect_evidence.ps1`.  
- **SQLite** (stdlib): generation history database file under `artifacts/db/` (default layout; see code and `.env.example`).

### Frontend

- **Vanilla HTML/JS** (no separate Node frontend build in this repository).

---

## 4. Repository layout

```
NovaLM-Hybrid/
  configs/           # model, train, infer, data, retrieval.yaml
  scripts/           # CLI: preprocess, tokenizer, train, eval, compare, scores, smoke, evidence, ...
  src/
    data_pipeline/   # ingest, clean, dedupe, split, chunk index helpers
    tokenization/    # SentencePiece training + wrapper
    training/        # train loop, evaluate, metrics
    models/          # transformer_lm, retriever, hybrid_generator
    serving/         # FastAPI api.py, schemas, db, web/
    utils/           # config, checkpoint, chunks_json, paths, ...
  tests/             # pytest (unit + @pytest.mark.integration for full API)
  artifacts/         # checkpoints, tokenizer, db, reports (gitignored pieces may apply)
  data/              # raw, processed, splits, index
  README.md          # quick start, commands
  PROJECT_GUIDE.md   # this file
  pyproject.toml
  Dockerfile
  docker-compose.yml
  .env.example
  .github/workflows/ci.yml
```

**Important paths (defaults):**

- Checkpoint: `artifacts/checkpoints/best.pt`  
- Tokenizer: `artifacts/tokenizers/tokenizer.model`  
- Retrieval index: `data/index/chunks.json`  
- Reports: `artifacts/reports/` (`final_report.md`, `compare_rows.json`, `scores.csv`, `evidence/`)

---

## 5. End-to-end data flow

```text
data/raw/*.txt
    → preprocess (clean, dedupe, splits)
    → train SentencePiece → tokenizer.model
    → build token IDs JSON
    → train Transformer LM → best.pt
    → build chunks.json (for TF-IDF)
    → serve FastAPI (load model + tokenizer + fit retriever on chunks)
         → /generate        (pure)
         → /hybrid_generate (retrieve top-k → generate)
         → history → SQLite
```

Anyone building a **similar** project usually copies this same skeleton; the differentiating pieces are **corpus**, **model size**, **retriever choice**, and **product features** (auth, billing, etc.).

---

## 6. API surface

Typical routes (see `src/serving/api.py` and `README.md` for exact payloads):

| Method | Path | Role |
|--------|------|------|
| GET | `/health` | Liveness; includes device string. |
| GET | `/` | Web UI (HTML). |
| POST | `/generate` | Pure LM generation. |
| POST | `/hybrid_generate` | Retrieval + generation. |
| GET | `/history` | Paginated history with filters. |
| DELETE | `/history/{id}` | Delete one record. |
| GET | `/history/export` | JSON or CSV export. |
| GET | `/retrieve_debug` | Inspect TF‑IDF hits only. |
| GET | `/docs` | OpenAPI (FastAPI automatic). |

---

## 7. Configuration and environment

- **YAML configs** in `configs/`: model, training, inference defaults, data paths, and **retrieval** (`retrieval.yaml`: TF‑IDF options, optional `min_context_score`, hybrid prompt template).  
- **Environment variables** (see `.env.example`):  
  - `NOVALM_CHECKPOINT`, `NOVALM_SP_MODEL`, `NOVALM_CHUNKS_JSON` — resolved in `src/utils/config.py`.  
  - **Compose-only:** `NOVALM_HOST_PORT` (see `docker-compose.yml`).  
- **`.env` loading:** When Python imports `src.utils.config`, **`python-dotenv`** loads project-root `.env` without overriding variables already set in the shell.

---

## 8. Running locally

**Install (Windows example):**

```powershell
cd path\to\NovaLM-Hybrid
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

**Train / pipeline (high level):** follow `README.md` sections *Data Preparation*, *Train*, *Evaluate*.

**API (helper script):**

```powershell
.\scripts\run_api.ps1
# or with reload:
.\scripts\run_api.ps1 --reload
```

**Smoke test (requires API already running in another terminal):**

```powershell
$env:PYTHONPATH='.'
python scripts/smoke_release.py --base-url http://127.0.0.1:8080
```

---

## 9. Docker

```bash
docker compose build
docker compose up
```

Compose mounts checkpoints, tokenizer, DB, and `chunks.json`; see `docker-compose.yml` and `.env.example`. First startup can take **minutes** on CPU while weights load.

---

## 10. Evaluation, reports, and evidence

### Static metrics (no server)

```powershell
$env:PYTHONPATH='.'
python scripts/run_eval.py
```

Prints validation and test loss/perplexity from `best.pt` and splits (see script for exact definitions).

### Compare run (regenerates report + JSON)

```powershell
$env:PYTHONPATH='.'
python scripts/run_compare.py
```

Writes **`artifacts/reports/compare_rows.json`** and refreshes **`artifacts/reports/final_report.md`**, merging prior **Metrics** / **Score Averages** / narrative sections when sensible.

### Heuristic scores (not a human rubric)

```powershell
$env:PYTHONPATH='.'
python scripts/proxy_scores.py
python scripts/score_report.py
```

Fills **`scores.csv`**, injects score columns into the markdown table, and updates **Score Averages** in the report. Replace with **manual** `scores.csv` if your course requires blind human scoring.

### JSON “evidence pack” (substitute for screenshots when allowed)

With API running:

```powershell
.\scripts\collect_evidence.ps1
```

Outputs under **`artifacts/reports/evidence/`** (health, generate, hybrid, history, retrieve_debug as JSON).

---

## 11. Testing and CI

| Command | Meaning |
|---------|---------|
| `pytest -m "not integration"` | Fast unit tests; **no** full model load (used in CI). |
| `pytest` | Full suite including **integration** tests that **import** the API (requires `best.pt`, tokenizer, `chunks.json` on disk). |

CI (GitHub Actions) runs the **non-integration** subset so the repository stays testable without large binary artifacts.

---

## 12. Typical failure modes and limits

- **Small LM + modest data** → odd fluency, repetition, or tokenization artifacts; quality improves with **more text** and **more training**.  
- **TF‑IDF** matches **lexical overlap**, not “meaning”; wrong corpus or shared stopwords can pull **off-domain** chunks (e.g. literary text for technical prompts).  
- **Hybrid** can be **slower** than pure (retrieval + generation). Latency numbers depend on **CPU/GPU**, `max_new_tokens`, and batching (single-user server here).  
- **Heuristic scores** in `proxy_scores.py` are **transparent proxies** only; they are not a substitute for a required human rubric unless your grader accepts them.

---

## 13. If you build something similar

Reuse this **proven stack pattern**:

1. PyTorch decoder-only LM.  
2. SentencePiece (or another subword tokenizer).  
3. YAML configs + one training entrypoint.  
4. FastAPI + Pydantic for serving.  
5. Start with **TF‑IDF or BM25** retrieval; upgrade to **dense embeddings** when needed.  
6. SQLite (or a file log) for history in student/portfolio scope.  
7. pytest + smoke script + Docker.

---

## 14. Checklist before submission or demo

- [ ] `README.md` **Evaluation Summary** matches your latest `run_eval.py` / `run_compare.py` if you retrained.  
- [ ] `artifacts/reports/final_report.md` reads coherently; replace heuristic scores if rubric demands human marks.  
- [ ] Evidence: run **`collect_evidence.ps1`** **or** capture required **screenshots** per course instructions.  
- [ ] `pytest -m "not integration"` passes; run full `pytest` locally if you ship `best.pt`.  
- [ ] Optional: push to GitHub so **CI** runs on your fork.

---

## Where to read next

| Need | File |
|------|------|
| Commands and curl examples | `README.md` |
| This full picture in one place | `PROJECT_GUIDE.md` (you are here) |
| API behavior and schemas | `src/serving/api.py`, `src/serving/schemas.py` |
| Model and training | `src/models/transformer_lm.py`, `src/training/train.py` |
| Retrieval and hybrid path | `src/models/retriever.py`, `src/models/hybrid_generator.py` |

---

*End of project guide.*
