"""Minimal local API for the SiteSeer synthetic-data RAG demo."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from backend.integration.filtering import select_eligibility_client

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    from backend.llm import rag_pipeline

    app.state.rag = rag_pipeline
    app.state.embedding_model = rag_pipeline.SentenceTransformer(rag_pipeline.MODEL_NAME)
    from backend.integration.team2a import Team2AClient
    mode = os.getenv("TEAM2A_MODE", "disabled")
    if mode not in {"disabled", "api"}:
        raise RuntimeError("TEAM2A_MODE must be disabled or api.")
    app.state.eligibility_client = Team2AClient()
    app.state.filtering_enabled = mode == "api"
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


class ChatMessage(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=6)
    stage: str | None = None
    trade: str | None = None
    jurisdiction: str | None = None
    learner_level: str | None = None
    context: "QueryContext"


class QueryContext(BaseModel):
    """Provisional fields from the meeting; final 1A/2A contract pending."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    module_id: str = Field(min_length=1, max_length=200)
    stage_id: str = Field(min_length=1, max_length=200)
    siteseer_node_id: str = Field(min_length=1, max_length=200)
    selected_node_ids: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(max_length=100)
    jurisdiction: str = Field(min_length=1, max_length=200)
    trade_or_domain: str = Field(min_length=1, max_length=200)
    learner_level: str = Field(min_length=1, max_length=200)


QuestionRequest.model_rebuild()


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
        eligibility_client = select_eligibility_client(
            request.app.state.eligibility_client,
            request.app.state.filtering_enabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        connection = rag.get_database_connection()
        try:
            return rag.answer_question(
                **body.model_dump(exclude={"context"}),
                query_context=body.context.model_dump(),
                eligibility_client=eligibility_client,
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
