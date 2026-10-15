# SiteSeer React frontend

A single-page React interface for the local FastAPI RAG backend. It sends only
`question` to `/ask` and displays the answer, refusal/error states and cited EKO
titles. Citations link to the corresponding source cards. No filters or chat
history are sent. Model output is displayed as plain text, not rendered HTML.

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
The page waits up to three minutes; timing out does not cancel server generation.

The project uses React and Vite 6, compatible with the existing Node 20.13 setup.
The API defaults to http://127.0.0.1:8000. To change it, create `frontend/.env`
with `VITE_API_URL=http://127.0.0.1:8000` and restart Vite.

```bash
npm run build
```

Builds the app into `dist/`. `npm run preview` serves that build on port 5173
(the backend still needs to be running). Stop the dev server first before previewing.
