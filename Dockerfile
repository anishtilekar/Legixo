# Python 3.11 is pinned deliberately: pinecone and langchain-together cap support
# at 3.13, and 3.14 hits wheel build failures. This image exists so a reviewer
# with a different Python can run the project without touching their system.
FROM python:3.11-slim

WORKDIR /app

# Dependencies first so code edits don't invalidate the pip layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY corpus/ ./corpus/
COPY eval/ ./eval/

# .env is NOT copied — it is gitignored and holds real keys. Pass configuration
# at run time with --env-file .env (see README).

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
