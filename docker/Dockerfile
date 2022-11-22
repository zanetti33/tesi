FROM conda/miniconda3

WORKDIR /home/zanetti

SHELL ["/bin/bash", "-c"]

RUN conda update conda \
    && conda create --name radiography_retrieval \
    && conda init bash \
    && source /root/.bashrc \
    && conda activate radiography_retrieval 

RUN conda install pytorch torchvision  cudatoolkit=11.3 -c pytorch -y

RUN conda install -c huggingface transformers -y \
    && conda install -c anaconda pandas -y \
    && conda install -c huggingface -c conda-forge datasets -y \
    && conda install -c conda-forge scikit-learn -y \
    && conda install -c conda-forge wandb=0.13.5 -y\
    && conda install -c metric-learning pytorch-metric-learning=1.6.3 -y\
    && conda install -c conda-forge tensorboard=2.10.1 -y

RUN pip install record-keeper==0.9.32
