FROM python:3.11-slim

WORKDIR /backend

RUN pip install --no-cache-dir fastapi uvicorn Pillow

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]