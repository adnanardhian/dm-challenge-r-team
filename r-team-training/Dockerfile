# Docker image for training

FROM ubuntu

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    time \
  && rm -rf /var/lib/apt/lists/*

RUN curl -qsSLkO \
    https://repo.continuum.io/miniconda/Miniconda-latest-Linux-`uname -p`.sh \
  && bash Miniconda-latest-Linux-`uname -p`.sh -b \
  && rm Miniconda-latest-Linux-`uname -p`.sh

ENV PATH=/root/miniconda2/bin:$PATH

RUN conda install -y \
    h5py \
    pandas \
    scikit-learn \
    tensorflow \
    pytables \
  && conda clean --yes --tarballs --packages --source-cache \
  && pip install --upgrade pip \
  && pip install --upgrade -I setuptools \
  && pip install git+git://github.com/fchollet/keras.git \
  && pip install --upgrade -I setuptools \
  && pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0rc0-cp27-none-linux_x86_64.whl

# copy training and test files
RUN echo "copy training & test files.."
COPY train.py /train.py
COPY train.sh /train.sh

RUN echo "chmod training & test files.."
RUN chmod 777 /*.py
RUN chmod 777 /*.sh

RUN echo "complete."
