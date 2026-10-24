# SiteSeer React frontend

A single-page React interface for the local FastAPI RAG backend. It sends the current
`question` and up to three recent question/answer exchanges to `/ask` and displays the answer, refusal/error states and cited EKO
titles. Citations link to the corresponding source cards. No filters are sent. Conversation history stays in React memory and is cleared
on refresh or by the New chat button. Nothing is saved to browser storage. Model output is displayed as plain text, not rendered HTML.

## Run locally

Keep PostgreSQL and Ollama running. Start the backend from the repository root,
in your working Python environment:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173. Enter a question or select an example, then click
**Ask SiteSeer**. The first answer can take longer while Ollama loads the model.
The page waits up to five minutes; timing out does not cancel server generation.

The project uses React and Vite 6, compatible with the existing Node 20.13 setup.
The API defaults to http://127.0.0.1:8000. To change it, create `frontend/.env`
with `VITE_API_URL=http://127.0.0.1:8000` and restart Vite.

```bash
npm run build
```

Builds the app into `dist/`. `npm run preview` serves that build on port 5173
(the backend still needs to be running). Stop the dev server first before previewing.


## Follow-up questions

Ask an initial question, then refer to it naturally (for example, “What evidence
should I record for that?”). The backend uses recent messages to resolve the
question before retrieving fresh EKO evidence. Older messages remain visible,
but only the last three exchanges are sent; very long prior answers are limited
to 6,000 characters. Follow-ups take an extra model call and may be slower.
Failed requests preserve previous turns and restore the question for retry.
New chat is disabled while an answer is being generated.
