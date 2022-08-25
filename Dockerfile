# syntax=docker/dockerfile:1

FROM python:3.9-slim

WORKDIR /app

EXPOSE 5000

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY main.py main.py
COPY auto_seq_model_glass_trained.h5 auto_seq_model_glass_trained.h5
COPY auto_seq_model_cogen_trained.h5 auto_seq_model_cogen_trained.h5
COPY mse_glass.npy mse_glass.npy
COPY mse_cogen.npy mse_cogen.npy



CMD python main.py
