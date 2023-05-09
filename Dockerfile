# syntax=docker/dockerfile:1

FROM python:3.10-bullseye

EXPOSE 7865

WORKDIR /app

COPY . .

RUN pip3 install poetry
RUN poetry install
RUN poetry shell

CMD ["uvicorn", "train-api:app", "--host", "0.0.0.0", "--port", "7865"]