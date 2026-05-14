from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    max_new_tokens: int | None = Field(default=None, ge=1, le=256)
    temperature: float | None = Field(default=None, ge=0.1, le=2.0)
    top_k: int | None = Field(default=None, ge=1, le=200)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    repetition_penalty: float | None = Field(default=None, ge=1.0, le=2.0)

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt cannot be blank.")
        return v.strip()


class GenerateResponse(BaseModel):
    generated_text: str
    latency_ms: float


class HybridGenerateRequest(GenerateRequest):
    retrieval_top_k: int | None = Field(default=None, ge=1, le=10)


class HybridGenerateResponse(BaseModel):
    contexts: List[dict]
    generated_text: str
    latency_ms: float


class HistoryRecord(BaseModel):
    id: int
    mode: str
    prompt: str
    generated_text: str
    contexts: List[dict]
    latency_ms: float
    created_at: str
