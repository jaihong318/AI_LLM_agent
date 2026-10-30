FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=True \
    PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8080

CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
