FROM python:3.10.11-slim

EXPOSE 7865
RUN mkdir -p /app
WORKDIR /app

COPY . .

RUN poetry export -f requirements.txt --output requirements.txt
RUN apt update && apt install build-essential
RUN pip install -r requirements.txt

CMD ["python3", "train-api.py"]