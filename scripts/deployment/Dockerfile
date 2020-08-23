FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

WORKDIR /usr/

RUN pip install matplotlib
RUN pip install seaborn
RUN pip install tensorboard
RUN pip install pandas
RUN conda install -c conda-forge umap-learn

RUN mkdir /usr/numba_cache
RUN chmod 777 /usr/numba_cache
ENV NUMBA_CACHE_DIR=/usr/numba_cache
EXPOSE 6006
