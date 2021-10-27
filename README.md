#FSE_project

#KDnet
This repository is an independent implementation of https://arxiv.org/abs/1704.01222 in pytorch.

#Quickstart

1. Type "git clone https://github.com/Victor11215546478/FSE_project.git".
2. download:
  sh download.sh

3. build:
  docker build -t kdnet .

4. run:
  docker run -it kdnet
4. Type "python3 train.py" to train network.
5. Type "python3 test.py" to test network.
6. Type "python3 -m unittest critical_test.py" to run unittest.
#Development
To develop this network you can optimize number of layers, iterations, change optmizer, neural network's structure.