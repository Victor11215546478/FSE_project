# KDnet
This repository is an independent implementation of https://arxiv.org/abs/1704.01222 in pytorch.

## Disclaimer
This repository is not an original official implementation of the work, but a refactored codebase. Performed within the FSE coursework at Skoltech.

## General idea

Authors present a new deep learning architecture (called Kdnetwork) that is designed for 3D model recognition tasks and works with unstructured point clouds. The new architecture performs multiplicative transformations and shares parameters of these transformations according to the subdivisions of the point clouds imposed onto them by kdtrees.
Kd-networks demonstrate competitive performance in a number of shape recognition tasks such as shape classification, shape retrieval and shape part segmentation.

## Quickstart

0) Launch Docker on your local machine
1) Create a directory for the project and navigate there running following command in terminal: `mkdir <FOLDER> && cd <FOLDER>`
2) Clone repository on your local machine `git clone https://github.com/SergeyPetrakov/kdnet.pytorch && cd kdnet.pytorch`
3) Run `make build` in terminal to build the docker image for project
4) Run `make run` to launch docker container in an interactive mode

   * Network is trained using the command `python3 train.py`
   * Test will run with the command `python3 test.py`
   * Unitest will run with the command `python3 -m unittest critical_test.py`


## Super quickstart

1) Run `docker pull kovanic1998/kdnet.torch` to pull image of the project from Docker Hub
2) Run `docker run -it --name kdnet_container kovanic1998/kdnet.torch` to launch docker container in an interactive mode

   * Network is trained using the command `python3 train.py`
   * Test will run with the command `python3 test.py`
   * Unitest will run with the command `python3 -m unittest critical_test.py`

## Development

File `train.py` contain  neural network training. One of the possible ways to develop this project is to optimize number of iterations, layers, change optmizer, structure of neural network

## Notes

1) You can check the critical functionality of the modules of the project using `critical_test.py` file
2) Deployment system based on GitHub workflows is also provided. Use `main.yml` file and go to `Actions`
3) Link to docker image on DockerHub: `https://hub.docker.com/repository/docker/kovanic1998/kdnet.torch`
