# AMR_anomaly
calculate health score for turbine meter in NGR AMR system with seqence auto encoder

requirement:

keras 2.3.1

tensorflow 2.0.0

flask 2.0.0

numpy 1.16.4

Creating the function that take 7 sequence of 1 feature (natural gas volume in MMSCFD) from turbine meter and calculate percentile of anomaly as an output
================================================================================= REST API Server Deployment

Requires:

run_keras_server.py

auto_seq_model_glass_trained.h5

auto_seq_model_cogen_trained.h5

mse_glass.npy

mse_cogen.npy

======================================================

Implementation :

run run_keras_server.py on server

Client can POST request with

URL : server.ip/predict

content : {'Date':str,'Ship to':str,'seq':list }

'Date'      : sample date in YYYY-mm-dd Type : str

'Ship to'   : Ship to code  Type : str

'seq'       : list of 7 sequence data (current date first)  Type : list of float

eg. {'Date': '2021-01-05', 'Ship to': '30019567', 'seq': [2.7007623, 2.7153472, 2.7237521, 2.7370658, 2.7226574, 2.7027399, 2.7077193]}

Response : {'Date': str, 'Ship to': str, 'percentile': float, 'success': bool}

'Date': sample date in YYYY-mm-dd Type : str

'Ship to': Ship to code  Type : str

'percentile': percentile of mse of sample indicate chance to be anomaly sequence  type : float 

'success': ensure the request was successful  type : bool

eg. {'Date': '2021-01-05', 'Ship to': '30019567', 'percentile': 0.06748992934028925, 'success': True}
