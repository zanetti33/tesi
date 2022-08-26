FROM conda/miniconda3

WORKDIR /home/zanetti

SHELL ["/bin/bash", "-c"]

RUN conda update conda \
    && conda create --name radiography_retrieval \
    && conda init bash \
    && source /root/.bashrc \
    && conda activate radiography_retrieval \
    && conda install pytorch torchvision  cudatoolkit=11.3 -c pytorch -y \
    && conda install -c huggingface transformers -y \
    && conda install -c anaconda pandas -y \
    && conda install -c huggingface -c conda-forge datasets -y \
    && conda install -c conda-forge scikit-learn -y