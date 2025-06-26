FROM python:3.9
RUN apt-get update && apt-get install -y git libgl1-mesa-glx

WORKDIR /app

COPY requirements.txt .
COPY external/ external/

RUN pip install torch torchvision
RUN pip install -r requirements.txt
RUN pip install pillow==9.5.0

COPY . .


CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]