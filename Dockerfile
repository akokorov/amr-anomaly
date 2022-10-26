# syntax=docker/dockerfile:1

#FROM python:3.9
FROM tadeorubio/pyodbc-msodbcsql17

WORKDIR /app

EXPOSE 5000

COPY requirements.txt requirements.txt

COPY main.py main.py
COPY auto_seq_model_glass_trained.h5 auto_seq_model_glass_trained.h5
COPY auto_seq_model_cogen_trained.h5 auto_seq_model_cogen_trained.h5
COPY mse_glass.npy mse_glass.npy
COPY mse_cogen.npy mse_cogen.npy

# install FreeTDS and dependencies
#RUN apt-get update \
# && apt-get install unixodbc -y \
# && apt-get install unixodbc-dev -y \
# && apt-get install freetds-dev -y \
# && apt-get install freetds-bin -y \
# && apt-get install tdsodbc -y \
# && apt-get install --reinstall build-essential -y

#populate "ocbcinst.ini" as this is where ODBC driver config sits
#RUN echo "[FreeTDS]\n\
#Description = FreeTDS Driver\n\
#Driver = /usr/lib/x86_64-linux-gnu/odbc/libtdsodbc.so\n\
#Setup = /usr/lib/x86_64-linux-gnu/odbc/libtdsS.so" >> /etc/odbcinst.ini

RUN pip install -r requirements.txt
RUN pip install sqlalchemy

CMD python main.py
