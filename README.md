#FSE_project
# _KDNet_
This repository contains the refactored database with an implementation of https://arxiv.org/abs/1704.01222 in pytorch. 

# _Annotation_
Authors of the article above developed a new deep learning architecture for  3D model recognition tasks called Kd-network. This architecture is based on kd-trees and gives certain advantage compared to the currently dominant convolutional architectures. Authors' technique provides remarkably good perfomance on a variety of popular benchmarks 

#  _Quickstart_
1. Type "git clone https://github.com/Victor11215546478/FSE_project.git".
2. download:
  sh download.sh
3. build:
  docker build -t kdnet .
4. run:
  docker run -it kdnet
5. Type "python3 train.py" to train network.
6. Type "python3 test.py" to test network.
7. Type "python3 -m unittest critical_test.py" to run unittest.

#Development
To develop this network you can optimize number of layers, iterations, change optmizer, neural network's structure.
