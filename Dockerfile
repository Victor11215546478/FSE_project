FROM python:3.8

RUN apt-get update --fix-missing && \
    apt-get -yq install wget unzip && \
    apt-get -yq install ffmpeg libsm6 libxext6

RUN mkdir "/repo"
WORKDIR "/repo"

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN ["sh","download.sh"]
RUN ["make"]

ENTRYPOINT ["/bin/bash"]
