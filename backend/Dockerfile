FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ backend/
COPY data/processed/search_ready_ekos.json data/processed/search_ready_ekos.json

RUN useradd --create-home siteseer \
    && mkdir -p data/embeddings \
    && chown siteseer data/embeddings
USER siteseer

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]