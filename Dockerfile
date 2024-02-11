FROM python:3.7.17-slim-bookworm

RUN apt-get update && apt-get -f install && apt-get install -y python3-pip

RUN mkdir /opt/app

COPY . /opt/app

WORKDIR /opt/app

RUN pip3 install -r requirements.txt

CMD ["python","app.py"]


