FROM python:3.11-slim

WORKDIR /tg_bot

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY logic.py .

CMD ["python", "logic.py"]
