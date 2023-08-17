FROM python:3.10.11-slim-buster AS base

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/
COPY storage/ /app/storage/
EXPOSE 80

ENV FLASK_APP=app.py

CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]

