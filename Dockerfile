FROM ubuntu:20.04

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \ 
    gfortran \
    make \
    libarpack2-dev \
    libnetcdf-dev \
    libnetcdff-dev \
    git \
    ca-certificates \
    openssh-client \
  && apt-get autoremove -y \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Aida-Alvera/DINEOF \
  && cd DINEOF/ \
  && cp config.mk.template config.mk \
  && make \
  && ln ./dineof /usr/local/bin/dineof

WORKDIR /home/dineof/