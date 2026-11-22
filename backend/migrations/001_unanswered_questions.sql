BEGIN;

CREATE TABLE IF NOT EXISTS unanswered_questions (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    question TEXT NOT NULL,
    topics TEXT[] NOT NULL,
    keywords TEXT[] NOT NULL,
    filters JSONB NOT NULL DEFAULT '{}'::jsonb,
    reason TEXT NOT NULL CHECK (
        reason IN ('no_matching_evidence', 'insufficient_support')
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS unanswered_questions_created_at_idx
    ON unanswered_questions (created_at DESC, id DESC);

CREATE INDEX IF NOT EXISTS unanswered_questions_topics_idx
    ON unanswered_questions USING GIN (topics);

COMMIT;