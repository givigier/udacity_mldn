FROM continuumio/anaconda

RUN /opt/conda/bin/conda install jupyter -y --quiet

RUN mkdir -p /ml

WORKDIR /ml

COPY . .

EXPOSE 8888
