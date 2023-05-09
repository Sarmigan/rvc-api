FROM python:3.10.11-slim

EXPOSE 7865
RUN mkdir -p /app
WORKDIR /app

COPY . .

RUN apt update && apt install build-essential -y
RUN pip install poetry
RUN poetry export -f requirements.txt --output requirements.txt
RUN pip install -r requirements.txt

CMD ["python", "train-api.py", "--port", "7865"]