FROM python:3.8 

RUN mkdir "/repo"
WORKDIR "/repo"

RUN apt-get update && \
    apt-get -yq install wget unzip ffmpeg libsm6 libxext6

COPY . .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN ["sh", "download.sh"]
RUN ["make"]

ENTRYPOINT ["/bin/bash"]
