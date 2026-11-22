"""Persistence and listing operations for unanswered questions."""

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def save_unanswered_question(
    connection,
    *,
    question: str,
    topics: list[str],
    keywords: list[str],
    filters: dict,
    reason: str,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO unanswered_questions (
                question, topics, keywords, filters, reason
            )
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                question,
                topics,
                keywords,
                Jsonb(filters),
                reason,
            ),
        )

    # Closing the connection without this would roll back the insert.
    connection.commit()


def list_unanswered_questions(
    connection,
    *,
    limit: int,
    offset: int,
    topic: str | None = None,
) -> dict:
    # SQL fragments below are fixed strings; user input remains parameterized.
    where_clause = ""
    parameters = []

    if topic is not None:
        where_clause = "WHERE topics @> %s::text[]"
        parameters.append([topic])

    # One statement provides a consistent total and page, including empty pages.
    sql = f"""
        WITH filtered AS (
            SELECT
                id, question, topics, keywords,
                filters, reason, created_at
            FROM unanswered_questions
            {where_clause}
        ),
        page AS (
            SELECT *
            FROM filtered
            ORDER BY created_at DESC, id DESC
            LIMIT %s OFFSET %s
        )
        SELECT
            (SELECT COUNT(*) FROM filtered) AS total,
            page.*
        FROM (SELECT 1) AS anchor
        LEFT JOIN page ON TRUE
        ORDER BY page.created_at DESC, page.id DESC
    """

    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute(sql, [*parameters, limit, offset])
        rows = cursor.fetchall()

    return {
        "items": [
            {
                **{key: value for key, value in row.items() if key != "total"},
                # Older records included conversation history as a filter.
                "filters": {
                    key: value for key, value in row["filters"].items()
                    if key != "history"
                },
            }
            for row in rows
            if row["id"] is not None
        ],
        "total": rows[0]["total"],
        "limit": limit,
        "offset": offset,
    }


def list_unanswered_topics(connection) -> dict:
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT
                topic,
                COUNT(DISTINCT q.id) AS question_count
            FROM unanswered_questions AS q
            CROSS JOIN LATERAL unnest(q.topics) AS t(topic)
            GROUP BY topic
            ORDER BY question_count DESC, topic ASC
            """
        )
        items = cursor.fetchall()

    return {"items": items}
