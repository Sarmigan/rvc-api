FROM python:3.10.11-bullseye

EXPOSE 7865

WORKDIR /app

COPY . .

RUN pip3 install poetry
RUN apt update && apt install build-essential
RUN poetry install

CMD ["python3", "poetry", "run", "train-api.py"]