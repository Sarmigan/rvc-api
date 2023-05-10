FROM python:3.10.11-slim

EXPOSE 7865
RUN mkdir -p /app
WORKDIR /app

COPY . .