"""Minimal local API for the SiteSeer synthetic-data RAG demo."""

import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime

from backend.question_classification import classify_question
from backend.unanswered_questions import (
    list_unanswered_questions,
    list_unanswered_topics,
    save_unanswered_question,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from backend.llm import rag_pipeline

    app.state.rag = rag_pipeline
    app.state.embedding_model = rag_pipeline.RemoteEmbeddingModel(
        rag_pipeline.MODEL_NAME
    )
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


class UnansweredQuestionResponse(BaseModel):
    id: int
    question: str
    topics: list[str]
    keywords: list[str]
    filters: dict[str, str]
    reason: Literal["no_matching_evidence", "insufficient_support"]
    created_at: datetime


class UnansweredQuestionListResponse(BaseModel):
    items: list[UnansweredQuestionResponse]
    total: int
    limit: int
    offset: int


class UnansweredTopicResponse(BaseModel):
    topic: str
    question_count: int


class UnansweredTopicListResponse(BaseModel):
    items: list[UnansweredTopicResponse]

@app.get("/health")
def health():
    """API liveness only; does not check PostgreSQL or Ollama."""
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(body: QuestionRequest, request: Request):
    """Answer the question and save requests with insufficient evidence."""
    rag = request.app.state.rag

    try:
        connection = rag.get_database_connection()

        try:
            result = rag.answer_question(
                **body.model_dump(),
                embedding_model=request.app.state.embedding_model,
                connection=connection,
            )

            if result["status"] == "insufficient_evidence":
                topics, keywords = classify_question(body.question)

                filters = body.model_dump(
                    exclude={"question"},
                    exclude_none=True,
                )

                reason = (
                    "no_matching_evidence"
                    if not result["retrieval"]
                    else "insufficient_support"
                )

                save_unanswered_question(
                    connection,
                    question=body.question,
                    topics=topics,
                    keywords=keywords,
                    filters=filters,
                    reason=reason,
                )

            return result
        finally:
            connection.close()

    except Exception as exc:
        logger.exception("Question processing or persistence failed")
        raise HTTPException(
            status_code=503,
            detail="Unable to complete the question request right now.",
        ) from exc
@app.get(
    "/unanswered-questions/topics",
    response_model=UnansweredTopicListResponse,
)
def get_unanswered_topics(request: Request):
    """List distinct topics and their unanswered-question counts."""
    rag = request.app.state.rag

    try:
        connection = rag.get_database_connection()
        try:
            return list_unanswered_topics(connection)
        finally:
            connection.close()
    except Exception as exc:
        logger.exception("Unable to list unanswered topics")
        raise HTTPException(
            status_code=503,
            detail="Unable to retrieve unanswered topics right now.",
        ) from exc

@app.get(
    "/unanswered-questions",
    response_model=UnansweredQuestionListResponse,
)
def get_unanswered_questions(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    topic: str | None = Query(default=None, min_length=1, max_length=100),
):
    """List unanswered questions, optionally filtered by topic."""
    normalized_topic = None

    if topic is not None:
        normalized_topic = " ".join(topic.casefold().split())

        if not normalized_topic:
            raise HTTPException(
                status_code=422,
                detail="Topic cannot be blank.",
            )

    rag = request.app.state.rag

    try:
        connection = rag.get_database_connection()
        try:
            return list_unanswered_questions(
                connection,
                limit=limit,
                offset=offset,
                topic=normalized_topic,
            )
        finally:
            connection.close()
    except Exception as exc:
        logger.exception("Unable to list unanswered questions")
        raise HTTPException(
            status_code=503,
            detail="Unable to retrieve unanswered questions right now.",
        ) from exc