FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY models/ ./models/

ENV MODEL_PATH=/app/models/loan_approval_pipeline.joblib
ENV PORT=8080

EXPOSE 8080

# Use shell form or ${PORT} environment variable dynamically
CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}