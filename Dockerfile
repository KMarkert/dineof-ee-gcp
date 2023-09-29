# Use an official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.10-slim

ARG DEBIAN_FRONTEND=noninteractive

# install deps for DINEOF
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \ 
    gfortran \
    make \
    libarpack2-dev \
    netcdf-bin \
    libnetcdf-dev \
    libnetcdff-dev \
    git \
    ca-certificates \
    openssh-client \
  && apt-get autoremove -y \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# install DINEOF
RUN git clone https://github.com/Aida-Alvera/DINEOF \
  && cd DINEOF/ \
  && cp config.mk.template config.mk \
  && make \
  && ln ./dineof /usr/local/bin/dineof


COPY ./requirements.txt .
# Install Python dependencies.
RUN pip install -r requirements.txt

# Copy local code to the container image.
WORKDIR /src
COPY ./src/ .

WORKDIR /home/dineof/

ENTRYPOINT python /src/main.py