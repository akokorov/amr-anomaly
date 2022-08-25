# lstm autoencoder recreate sequence
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
import pandas as pd
# define input sequence
#X1 = pd.read_csv('seq_data_a.csv',index_col=0)
#X2 = pd.read_csv('seq_data_b.csv',index_col=0)

#x = np.dstack((np.array(X1), np.array(X2)))

X = pd.read_csv('seq_data_cogen.csv',index_col=0)[['seq01', 'seq02', 'seq03', 'seq04', 'seq05', 'seq06', 'seq07']]

samples = len(X)
timesteps = X.shape[1]
#features = X.shape[2]
features = 1

# reshape input into [samples, timesteps, features]
#sequence = sequence.reshape((samples, timesteps, features))
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(timesteps,features)))
model.add(RepeatVector(timesteps))
model.add(LSTM(100, activation='relu', return_sequences=True))
#model.add(TimeDistributed(Dense(features)))
model.add(TimeDistributed(Dense(32)))
model.add(TimeDistributed(Dense(8)))
model.add(TimeDistributed(Dense(features)))
model.compile(optimizer='adam', loss='mse')

#plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
model.summary()
model.save('auto_seq_model_dense7.h5')