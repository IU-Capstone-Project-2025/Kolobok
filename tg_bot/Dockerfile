FROM python:3.11-slim

WORKDIR /tg_bot

RUN pip install -r requirements.txt

COPY logic.py .

CMD ["python", "logic.py"]
