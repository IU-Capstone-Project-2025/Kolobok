FROM python:3.11-slim

WORKDIR /tg_bot

RUN pip install --no-cache-dir python-telegram-bot

COPY hello.py .

CMD ["python", "hello.py"]