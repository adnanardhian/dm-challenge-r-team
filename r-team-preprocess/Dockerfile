FROM ubuntu

MAINTAINER Diah Anggraeni Pitaloka <diahpitaloka15@gmail.com>

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	imagemagick \
	parallel \
	python-pip
	
RUN apt-get install -y python-numpy \
	python-pandas

WORKDIR /
COPY generate_image_labels.py .
COPY preprocess.sh .
