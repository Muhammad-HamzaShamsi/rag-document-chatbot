FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 CHROMA_DIR=/data/chroma_db
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY rag/ rag/
COPY app/ app/
COPY data/ data/
RUN useradd --create-home appuser && mkdir -p /data && chown appuser /data
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
