FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

COPY . .

RUN --mount=type=secret,id=envfile \
    sh -c 'set -e && set -a && . /run/secrets/envfile && set +a && python -m scripts.init'

CMD ["sh", "-c", "gunicorn app:app -k uvicorn.workers.UvicornWorker -w 4 --preload --bind 0.0.0.0:8000 --timeout 300 --graceful-timeout 300"]
