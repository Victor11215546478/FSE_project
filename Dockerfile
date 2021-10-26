FROM ubuntu:20.04

ENV TZ=Europe/Moscow

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update --fix-missing && \
    apt-get -yq install python3-pip wget unzip && \
    apt-get -yq install ffmpeg libsm6 libxext6


RUN mkdir "/project"
WORKDIR "/project"

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN ["sh","download.sh"]
RUN ["make", "render-balls"]