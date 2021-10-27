#FSE_project
# _KDNet_
This repository contains the refactored database with an implementation of https://arxiv.org/abs/1704.01222 in pytorch.  The original project can be found via the link https://github.com/fxia22/kdnet.pytorch


# _Annotation_
Authors of the article above developed a new deep learning architecture for  3D model recognition tasks called Kd-network. This architecture is based on kd-trees and gives certain advantage compared to the currently dominant convolutional architectures. Authors' technique provides remarkably good perfomance on a variety of popular benchmarks 

You can find a docker image via the below link:
https://hub.docker.com/repository/docker/victor11215546478/fse_project

#  _Quickstart_
1. Type "git clone https://github.com/Victor11215546478/FSE_project.git".
2. build:
  docker build -t kdnet .
3. run:
  docker run -it kdnet (see Note below)
4. Type "python3 train.py" to train network.
5. Type "python3 test.py" to test network.
6. Type "python3 -m unittest critical_test.py" to run unittest.


  
Note: If you downloaded the image from dockerhab, then you need to do sh download.sh after starting (after step 3) 

#Development
To develop this network you can optimize number of layers, iterations, change optmizer, neural network's structure.
