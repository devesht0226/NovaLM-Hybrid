from pathlib import Path
import csv
import io
import time

import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from src.models.hybrid_generator import HybridGenerator
from src.models.retriever import TfidfRetriever
from src.models.transformer_lm import MiniTransformerLM, TransformerConfig
from src.serving.db import GenerationDB
from src.serving.schemas import (
    GenerateRequest,
    GenerateResponse,
    HistoryRecord,
    HybridGenerateRequest,
    HybridGenerateResponse,
)
from src.utils.checkpoint import load_pt_checkpoint
from src.utils.chunks_json import load_chunks_json
from src.utils.config import get_path_from_env_or_default, load_all_configs

cfg = load_all_configs()
infer_cfg = cfg["infer"]
CHECKPOINT_PATH = get_path_from_env_or_default("NOVALM_CHECKPOINT", "artifacts/checkpoints/best.pt")
SP_MODEL_PATH = get_path_from_env_or_default("NOVALM_SP_MODEL", "artifacts/tokenizers/tokenizer.model")
CHUNKS_PATH = get_path_from_env_or_default("NOVALM_CHUNKS_JSON", "data/index/chunks.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
app = FastAPI(title="NovaLM-Hybrid API", version="1.2.0")
WEB_DIR = Path(__file__).resolve().parent / "web"
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
db = GenerationDB()


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": "internal_error", "message": str(exc)})


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Missing required {label}: {path}")


def load_model() -> MiniTransformerLM:
    _require_file(CHECKPOINT_PATH, "checkpoint")
    ckpt = load_pt_checkpoint(CHECKPOINT_PATH, map_location=DEVICE)
    cfg = TransformerConfig(**ckpt["config"])
    model = MiniTransformerLM(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_retriever_chunks(path: Path) -> list[str]:
    _require_file(path, "retrieval chunks")
    return load_chunks_json(path)


_require_file(SP_MODEL_PATH, "sentencepiece model")
model = load_model()
retrieval_cfg = cfg.get("retrieval") or {}
retriever = TfidfRetriever(retrieval_cfg)
chunks = load_retriever_chunks(CHUNKS_PATH)
retriever.fit(chunks)
_hybrid_instr = retrieval_cfg.get("hybrid_instruction")
if isinstance(_hybrid_instr, str) and not _hybrid_instr.strip():
    _hybrid_instr = None
generator = HybridGenerator(
    model=model,
    sp_model_path=str(SP_MODEL_PATH),
    retriever=retriever,
    device=DEVICE,
    hybrid_instruction=_hybrid_instr,
)


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    start = time.perf_counter()
    max_new_tokens = req.max_new_tokens if req.max_new_tokens is not None else int(infer_cfg.get("max_new_tokens", 80))
    temperature = req.temperature if req.temperature is not None else float(infer_cfg.get("temperature", 0.8))
    top_k = req.top_k if req.top_k is not None else int(infer_cfg.get("top_k", 40))
    top_p = req.top_p if req.top_p is not None else infer_cfg.get("top_p")
    top_p_f = float(top_p) if top_p is not None else None
    repetition_penalty = (
        req.repetition_penalty
        if req.repetition_penalty is not None
        else float(infer_cfg.get("repetition_penalty", 1.0))
    )
    text = generator.generate_pure(
        prompt=req.prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p_f,
        repetition_penalty=repetition_penalty,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    db.insert_generation(
        mode="pure",
        prompt=req.prompt,
        generated_text=text,
        contexts=[],
        latency_ms=latency_ms,
    )
    return {"generated_text": text, "latency_ms": round(latency_ms, 2)}


@app.post("/hybrid_generate", response_model=HybridGenerateResponse)
def hybrid_generate(req: HybridGenerateRequest):
    start = time.perf_counter()
    max_new_tokens = req.max_new_tokens if req.max_new_tokens is not None else int(infer_cfg.get("max_new_tokens", 80))
    temperature = req.temperature if req.temperature is not None else float(infer_cfg.get("temperature", 0.8))
    top_k = req.top_k if req.top_k is not None else int(infer_cfg.get("top_k", 40))
    retrieval_top_k = (
        req.retrieval_top_k if req.retrieval_top_k is not None else int(infer_cfg.get("retrieval_top_k", 3))
    )
    top_p = req.top_p if req.top_p is not None else infer_cfg.get("top_p")
    top_p_f = float(top_p) if top_p is not None else None
    repetition_penalty = (
        req.repetition_penalty
        if req.repetition_penalty is not None
        else float(infer_cfg.get("repetition_penalty", 1.0))
    )
    out = generator.generate_hybrid(
        prompt=req.prompt,
        retrieval_top_k=retrieval_top_k,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p_f,
        repetition_penalty=repetition_penalty,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    db.insert_generation(
        mode="hybrid",
        prompt=req.prompt,
        generated_text=out["generated_text"],
        contexts=out["contexts"],
        latency_ms=latency_ms,
    )
    return {"generated_text": out["generated_text"], "contexts": out["contexts"], "latency_ms": round(latency_ms, 2)}


@app.get("/retrieve_debug")
def retrieve_debug(q: str = Query(..., min_length=1, max_length=4000), k: int = Query(5, ge=1, le=20)):
    hits = retriever.top_k(q.strip(), k=k)
    return {"query": q.strip(), "k": k, "hits": [{"text": h.text, "score": h.score} for h in hits]}


@app.get("/history")
def history(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    mode: str | None = Query(None),
    q: str | None = Query(None),
):
    if mode and mode not in {"pure", "hybrid"}:
        raise HTTPException(status_code=400, detail="mode must be 'pure' or 'hybrid'")
    items = db.get_history(limit=limit, offset=offset, mode=mode, query=q)
    total = db.count_history(mode=mode, query=q)
    return {"total": total, "limit": limit, "offset": offset, "items": items}


@app.get("/history/export")
def history_export(
    fmt: str = Query("json", pattern="^(json|csv)$"),
    mode: str | None = Query(None),
    q: str | None = Query(None),
):
    rows = db.export_rows(mode=mode, query=q)
    if fmt == "json":
        return JSONResponse(content={"count": len(rows), "items": rows})

    out = io.StringIO()
    writer = csv.DictWriter(
        out,
        fieldnames=["id", "mode", "prompt", "generated_text", "contexts", "latency_ms", "created_at"],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return PlainTextResponse(
        content=out.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=history_export.csv"},
    )


@app.get("/history/{record_id}", response_model=HistoryRecord)
def history_by_id(record_id: int):
    row = db.get_by_id(record_id)
    if row is None:
        raise HTTPException(status_code=404, detail="History record not found")
    return row


@app.delete("/history/{record_id}")
def delete_history_record(record_id: int):
    deleted = db.delete_by_id(record_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="History record not found")
    return {"deleted": True, "id": record_id}


@app.delete("/history")
def delete_all_history():
    deleted_count = db.delete_all()
    return {"deleted": True, "count": deleted_count}
