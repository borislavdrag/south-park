FROM python:3.10-slim-buster

RUN apt-get update && \
    apt-get install -y unixodbc-dev gcc g++

COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt && \
    python -m spacy download en

ADD . /app

WORKDIR /app

EXPOSE 8000

CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]
