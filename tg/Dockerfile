FROM python:3.9-slim

WORKDIR /app

RUN pip install requests pillow python-telegram-bot

COPY main.py .

CMD ["python", "main.py"]
