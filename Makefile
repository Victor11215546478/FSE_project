render-balls:
	g++ -std=c++11 render_balls_so.cpp -o render_balls_so.so -shared -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 &&\
        rm render_balls_so.cpp

download:
	sh download.sh

build:
	docker build -t kdnet .

run:
	docker run -it kdnet

pull_and_run:
	docker pull kovanic1998/kdnet.torch
	docker run -it --name kdnet_container kovanic1998/kdnet.torch




