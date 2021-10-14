FROM python:3.9.5-slim-buster

COPY . .
RUN pip3 install -r requirements.txt --no-cache-dir

EXPOSE 80

CMD ["gunicorn", "-b", ":80", "app:app", "--timeout", "0"]

