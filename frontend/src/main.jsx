import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const API = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
const examples = [
  'What evidence should be recorded before a concrete slab pour?',
  'How do I stop a newly installed wall frame from moving?',
];
const headings = {
  answered: 'Your answer',
  insufficient_evidence: 'More knowledge is needed',
  invalid_model_response: 'An answer could not be verified',
  citation_validation_failed: 'An answer could not be verified',
};

function App() {
  const [question, setQuestion] = useState('');
  const [submitted, setSubmitted] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function ask(event) {
    event.preventDefault();
    if (!question.trim() || loading) return;
    setLoading(true);
    setError('');
    setResult(null);
    setSubmitted(question.trim());
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 180000);
    try {
      const response = await fetch(`${API}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question.trim() }),
        signal: controller.signal,
      });
      const data = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(typeof data?.detail === 'string'
          ? data.detail
          : 'The question could not be processed. Please try again.');
      }
      if (!data || typeof data.answer !== 'string' || !headings[data.status]
          || !Array.isArray(data.sources) || !Array.isArray(data.retrieval)) {
        throw new Error('The server returned an unexpected response. Please try again.');
      }
      setResult(data);
    } catch (err) {
      setError(err.name === 'AbortError'
        ? 'The answer is taking too long. Please try again shortly.'
        : err instanceof TypeError
          ? 'Cannot reach the backend. Check that FastAPI is running on port 8000.'
          : err.message);
    } finally {
      clearTimeout(timeout);
      setLoading(false);
    }
  }

  const sources = result ? result.sources.map(id => ({
    id,
    title: result.retrieval.find(item => item.ko_id === id)?.title || id,
  })) : [];

  function answerText(text) {
    // Render plain text safely; only known citations become source links.
    return text.split(/(\[MOCK-KO-\d+\])/g).map((part, index) => {
      const id = part.slice(1, -1);
      return sources.some(source => source.id === id)
        ? <a className="citation" key={index} href={`#source-${id}`}>{part}</a>
        : part;
    });
  }

  return (
    <div className="app">
      <header className="header">
        <a className="brand" href="/" aria-label="SiteSeer home"><span className="brand-mark" aria-hidden="true">S</span>SiteSeer<span className="brand-dot">.</span></a>
        <span className="header-caption">Knowledge assistant</span>
      </header>
      <main>
        <section className="intro">
          <p className="eyebrow">FROM KNOWLEDGE TO PRACTICE</p>
          <h1>A clearer answer.<br /><span>A better next step.</span></h1>
          <p className="intro-text">Ask a construction question. Explore an answer backed by the knowledge behind it.</p>
        </section>
        <section className="question-card" aria-label="Ask SiteSeer">
          <form onSubmit={ask}>
            <label htmlFor="question">What would you like to know?</label>
            <textarea id="question" value={question} onChange={e => setQuestion(e.target.value)}
              placeholder="For example, what should I check before a slab pour?"
              maxLength={2000} required disabled={loading} rows={4} />
            <div className="form-footer">
              <span className="helper">Be specific about the work or situation.</span>
              <button className="ask-button" disabled={loading || !question.trim()} type="submit">
                {loading ? 'Finding your answer…' : 'Ask SiteSeer'} <span aria-hidden="true">↗</span>
              </button>
            </div>
          </form>
        </section>
        {!result && !loading && !error && <section className="examples" aria-label="Example questions">
          <p>Not sure where to start? Try a question</p>
          <div>{examples.map(example => <button key={example} onClick={() => {
            setQuestion(example);
            document.getElementById('question').focus();
          }}>{example}<span aria-hidden="true">↗</span></button>)}</div>
        </section>}
        <div aria-live="polite" aria-busy={loading}>
          {loading && <section className="feedback loading"><span className="spinner" aria-hidden="true" /><div><strong>Looking through the knowledge</strong><p>Finding relevant sources and preparing your answer.</p></div></section>}
          {error && <section className="feedback error" role="alert"><strong>We couldn’t complete that request</strong><p>{error}</p></section>}
          {result && <section className="answer-card">
            <p className="eyebrow">{result.status === 'answered' ? 'KNOWLEDGE IN CONTEXT' : 'RESPONSE'}</p>
            <h2>{headings[result.status]}</h2>
            <p className="submitted">{submitted}</p>
            <div className="answer-text">{answerText(result.answer)}</div>
            {sources.length > 0 && <div className="sources"><h3>Referenced knowledge <span>{sources.length}</span></h3>
              {sources.map(source => <article tabIndex={-1} id={`source-${source.id}`} key={source.id} className="source"><span className="source-icon" aria-hidden="true">↳</span><div><strong>{source.title}</strong><p>{source.id}</p></div></article>)}
            </div>}
          </section>}
        </div>
        <footer>SiteSeer <span> / </span> Construction knowledge, connected.</footer>
      </main>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
