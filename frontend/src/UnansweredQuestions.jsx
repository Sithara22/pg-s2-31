import React, { useEffect, useState } from 'react';
import './unanswered-questions.css';

const API = (
  import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
).replace(/\/$/, '');

const PAGE_SIZE = 10;

const reasonLabels = {
  no_matching_evidence: 'No matching evidence',
  insufficient_support: 'Insufficient supporting evidence',
};

async function getJson(path, signal) {
  const response = await fetch(`${API}${path}`, { signal });
  const data = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(
      typeof data?.detail === 'string'
        ? data.detail
        : `Request failed (${response.status}).`
    );
  }

  return data;
}

export default function UnansweredQuestions() {
  const [topics, setTopics] = useState([]);
  const [topic, setTopic] = useState('');
  const [offset, setOffset] = useState(0);
  const [questions, setQuestions] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    const controller = new AbortController();

    async function load() {
      setLoading(true);
      setError('');
      setQuestions([]);

      const params = new URLSearchParams({
        limit: String(PAGE_SIZE),
        offset: String(offset),
      });

      if (topic) params.set('topic', topic);

      try {
        const [topicData, questionData] = await Promise.all([
          getJson('/unanswered-questions/topics', controller.signal),
          getJson(`/unanswered-questions?${params}`, controller.signal),
        ]);

        if (
          !Array.isArray(topicData?.items) ||
          !Array.isArray(questionData?.items) ||
          !Number.isInteger(questionData?.total)
        ) {
          throw new Error('The server returned an unexpected response.');
        }

        if (!controller.signal.aborted) {
          setTopics(topicData.items);
          setQuestions(questionData.items);
          setTotal(questionData.total);
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(
            err instanceof TypeError
              ? 'Cannot reach the backend. Check that FastAPI is running.'
              : err.message
          );
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }

    load();

    return () => controller.abort();
  }, [topic, offset, refresh]);

  function selectTopic(value) {
    setTopic(value);
    setOffset(0);
  }

  const first = total === 0 ? 0 : offset + 1;
  const last = Math.min(offset + questions.length, total);

  return (
    <div className="uq-page">
      <header className="header">
        <a className="brand" href="/" aria-label="SiteSeer home">
          <span className="brand-mark" aria-hidden="true">S</span>
          SiteSeer
          <span className="brand-dot">.</span>
        </a>

        <a
          href="/"
          className="header-caption"
          style={{ textDecoration: 'none' }}
        >
          Knowledge assistant ↗
        </a>
      </header>

      <main className="uq-main">
        <div className="uq-heading">
          <div>
            <p className="uq-eyebrow">KNOWLEDGE GAPS</p>
            <h1>Unanswered questions</h1>
            <p className="uq-description">
              Explore questions that need more supporting knowledge.
              Filter by topic to see where new content could help.
            </p>
          </div>

          <button
            type="button"
            className="uq-button"
            disabled={loading}
            onClick={() => setRefresh(value => value + 1)}
          >
            {loading ? 'Loading…' : 'Refresh'}
          </button>
        </div>

        <section className="uq-filters" aria-label="Filter by topic">
          <h2>Topics</h2>

          <div className="uq-topic-list">
            <button
              type="button"
              className={`uq-topic ${topic === '' ? 'is-active' : ''}`}
              aria-pressed={topic === ''}
              onClick={() => selectTopic('')}
            >
              All topics
            </button>

            {topics.map(item => (
              <button
                type="button"
                key={item.topic}
                className={`uq-topic ${
                  topic === item.topic ? 'is-active' : ''
                }`}
                aria-pressed={topic === item.topic}
                onClick={() => selectTopic(item.topic)}
              >
                {item.topic}
                <span>{item.question_count}</span>
              </button>
            ))}
          </div>
        </section>

        <section aria-label="Unanswered questions" aria-busy={loading}>
          <div className="uq-results-heading">
            <h2>{topic || 'All questions'}</h2>
            {!loading && !error && (
              <span>{total} {total === 1 ? 'question' : 'questions'}</span>
            )}
          </div>

          <div aria-live="polite">
            {loading && (
              <div className="uq-message">Loading unanswered questions…</div>
            )}

            {error && (
              <div className="uq-message uq-error" role="alert">
                <p>{error}</p>
                <button
                  type="button"
                  className="uq-button"
                  onClick={() => setRefresh(value => value + 1)}
                >
                  Try again
                </button>
              </div>
            )}

            {!loading && !error && questions.length === 0 && (
              <div className="uq-message">
                {total === 0
                  ? 'No unanswered questions found for this selection.'
                  : 'No questions on this page. Select Previous to go back.'}
              </div>
            )}

            {!loading && !error && questions.map(item => (
              <article className="uq-card" key={item.id}>
                <div className="uq-card-meta">
                  <span>Question #{item.id}</span>
                  <time dateTime={item.created_at}>
                    {new Date(item.created_at).toLocaleString()}
                  </time>
                </div>

                <h3>{item.question}</h3>

                <div className="uq-tags" aria-label="Question topics">
                  {item.topics.map(value => (
                    <button
                      type="button"
                      key={value}
                      onClick={() => selectTopic(value)}
                    >
                      {value}
                    </button>
                  ))}
                </div>

                <p className="uq-reason">
                  {reasonLabels[item.reason] || item.reason}
                </p>

                {item.keywords.length > 0 && (
                  <p className="uq-keywords">
                    <strong>Keywords:</strong> {item.keywords.join(', ')}
                  </p>
                )}

                {Object.keys(item.filters).length > 0 && (
                  <dl className="uq-metadata">
                    {Object.entries(item.filters).map(([key, value]) => (
                      <div key={key}>
                        <dt>{key.replaceAll('_', ' ')}</dt>
                        <dd>{value}</dd>
                      </div>
                    ))}
                  </dl>
                )}
              </article>
            ))}
          </div>

          {!loading && !error && (total > 0 || offset > 0) && (
            <nav className="uq-pagination" aria-label="Question pages">
              <button
                type="button"
                className="uq-button"
                disabled={offset === 0}
                onClick={() =>
                  setOffset(value => Math.max(0, value - PAGE_SIZE))
                }
              >
                Previous
              </button>

              <span>
                {questions.length > 0
                  ? `${first}–${last} of ${total}`
                  : `0 shown · ${total} total`}
              </span>

              <button
                type="button"
                className="uq-button"
                disabled={offset + PAGE_SIZE >= total}
                onClick={() => setOffset(value => value + PAGE_SIZE)}
              >
                Next
              </button>
            </nav>
          )}
        </section>
      </main>
    </div>
  );
}