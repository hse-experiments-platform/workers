FROM python:3.11

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN apt update
RUN apt install postgresql-client


COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]