FROM python:3.9.5-slim-buster

COPY . .
RUN pip3 install -r requirements.txt

EXPOSE 80

CMD ["python3", "app.py"]