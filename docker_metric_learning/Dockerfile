FROM ubuntu:20.04

ENV PATH="/root/miniconda3/bin:${PATH}"

ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get -y upgrade \
    && apt-get install -y --no-install-recommends wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda init bash \
    && . /root/.bashrc \
    && conda update conda -y  

RUN conda install -c conda-forge faiss wandb transformers

RUN conda install pytorch==1.12.1 cudatoolkit=11.3 -c pytorch

RUN conda install pytorch-metric-learning -c metric-learning -c pytorch

RUN conda install -c conda-forge tensorboard

RUN pip install record-keeper