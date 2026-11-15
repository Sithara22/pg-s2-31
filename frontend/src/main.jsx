import React, { useState, useRef, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const API = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
const examples = [
  'What evidence should be recorded before a concrete slab pour?',
  'How do I stop a newly installed wall frame from moving?',
];
const headings = {
  answered: 'Your answer',
  conversational: 'SiteSeer',
  insufficient_evidence: 'More knowledge is needed',
  invalid_model_response: 'An answer could not be verified',
  citation_validation_failed: 'An answer could not be verified',
};

function App() {
  const [question, setQuestion] = useState('');
  const [turns, setTurns] = useState([]);
  const endRef = useRef(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => { endRef.current?.scrollIntoView({ block: 'nearest' }); }, [turns, loading]);

  async function ask(event) {
    event.preventDefault();
    if (!question.trim() || loading) return;
    setLoading(true);
    setError('');
    const submitted = question.trim();
    const history = turns.filter(turn => turn.result && turn.result.status !== 'conversational').slice(-3).flatMap(turn => [
      { role: 'user', content: turn.question },
      { role: 'assistant', content: turn.result.answer.slice(0, 6000) },
    ]);
    setTurns(previous => [...previous, { question: submitted, result: null }]);
    setQuestion('');
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 420000);
    try {
      const response = await fetch(`${API}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: submitted, history }),
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
      setTurns(previous => previous.map((turn, index) => index === previous.length - 1 ? { ...turn, result: data } : turn));
    } catch (err) {
      setTurns(previous => previous.slice(0, -1));
      setQuestion(submitted);
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

  function renderTurn(turn, turnIndex) {
    const result = turn.result;
    const sources = result ? result.sources.map(id => ({
      id, title: result.retrieval.find(item => item.ko_id === id)?.title || id,
    })) : [];
    const answer = result?.answer.split(/(\[MOCK-KO-\d+\])/g).map((part, index) => {
      const id = part.slice(1, -1);
      return sources.some(source => source.id === id)
        ? <a className="citation" key={index} href={`#source-${turnIndex}-${id}`}>{part}</a>
        : part;
    });
    return <div className="chat-turn" key={turnIndex}>
      <section className="user-message"><span>You</span><p>{turn.question}</p></section>
      {result && <section className="answer-card">
        <p className="eyebrow">SITESEER</p>
        <h2>{headings[result.status]}</h2>
        <div className="answer-text">{answer}</div>
        {sources.length > 0 && <div className="sources"><h3>Referenced knowledge <span>{sources.length}</span></h3>
          {sources.map(source => <article tabIndex={-1} id={`source-${turnIndex}-${source.id}`} key={source.id} className="source"><span className="source-icon" aria-hidden="true">↳</span><div><strong>{source.title}</strong><p>{source.id}</p></div></article>)}
        </div>}
      </section>}
    </div>;
  }

  return (
    <div className="app">
      <header className="header">
        <a className="brand" href="/" aria-label="SiteSeer home"><span className="brand-mark" aria-hidden="true">S</span>SiteSeer<span className="brand-dot">.</span></a>
        <div className="header-actions"><span className="header-caption">Knowledge assistant</span>
          <button className="new-chat" disabled={loading} onClick={() => {
            setTurns([]); setQuestion(''); setError('');
            document.getElementById('question').focus();
          }}>New chat</button></div>
      </header>
      <main>
        {turns.length === 0 && <section className="intro">
          <p className="eyebrow">FROM KNOWLEDGE TO PRACTICE</p>
          <h1>A clearer answer.<br /><span>A better next step.</span></h1>
          <p className="intro-text">Ask a construction question. Explore an answer backed by the knowledge behind it.</p>
        </section>}
        <div className="conversation" aria-live="polite" aria-busy={loading}>
          {turns.map(renderTurn)}
          {loading && <section className="feedback loading"><span className="spinner" aria-hidden="true" /><div><strong>Preparing your answer</strong><p>Understanding your question and finding supporting knowledge.</p></div></section>}
          {error && <section className="feedback error" role="alert"><strong>We couldn’t complete that request</strong><p>{error}</p></section>}
          <div ref={endRef} />
        </div>
        <section className="question-card" aria-label="Ask SiteSeer">
          <form onSubmit={ask}>
            <label htmlFor="question">{turns.length ? 'Ask a follow-up question' : 'What would you like to know?'}</label>
            <textarea id="question" value={question} onChange={e => setQuestion(e.target.value)}
              placeholder="For example, what should I check before a slab pour?"
              maxLength={2000} required disabled={loading} rows={3} />
            <div className="form-footer">
              <span className="helper">Chat stays here until you refresh or start a new chat.</span>
              <button className="ask-button" disabled={loading || !question.trim()} type="submit">
                {loading ? 'Finding your answer…' : 'Ask SiteSeer'} <span aria-hidden="true">↗</span>
              </button>
            </div>
          </form>
        </section>
        {turns.length === 0 && !loading && !error && <section className="examples" aria-label="Example questions">
          <p>Not sure where to start? Try a question</p>
          <div>{examples.map(example => <button key={example} onClick={() => {
            setQuestion(example);
            document.getElementById('question').focus();
          }}>{example}<span aria-hidden="true">↗</span></button>)}</div>
        </section>}
        <footer>SiteSeer <span> / </span> Construction knowledge, connected.</footer>
      </main>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
