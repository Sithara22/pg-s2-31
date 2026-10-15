"""Minimal local API for the SiteSeer synthetic-data RAG demo."""

import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from backend.llm import rag_pipeline

    app.state.rag = rag_pipeline
    app.state.embedding_model = rag_pipeline.SentenceTransformer(rag_pipeline.MODEL_NAME)
    yield
    del app.state.embedding_model


app = FastAPI(
    title="SiteSeer Local Demo",
    description="Question answering using synthetic EKO data.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:3000", "http://127.0.0.1:3000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class QuestionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=2000)
    stage: str | None = None
    trade: str | None = None
    jurisdiction: str | None = None
    learner_level: str | None = None


class RetrievedSource(BaseModel):
    ko_id: str
    title: str
    score: float


class AnswerResponse(BaseModel):
    answer: str
    status: Literal[
        "answered", "insufficient_evidence",
        "invalid_model_response", "citation_validation_failed",
    ]
    sources: list[str]
    retrieval: list[RetrievedSource]


@app.get("/health")
def health():
    """API liveness only; does not check PostgreSQL or Ollama."""
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(body: QuestionRequest, request: Request):
    """Run the existing RAG pipeline with its default top-three retrieval."""
    rag = request.app.state.rag
    try:
        connection = rag.get_database_connection()
        try:
            return rag.answer_question(
                **body.model_dump(),
                embedding_model=request.app.state.embedding_model,
                connection=connection,
            )
        finally:
            connection.close()
    except Exception as exc:
        logger.exception("RAG request failed")
        raise HTTPException(
            status_code=503,
            detail="Unable to answer right now. Check PostgreSQL and Ollama, then try again.",
        ) from exc
