import pandas as pd
import numpy as np
#import keras
#from keras.models import Model,Input, load_model
#rom keras.layers import Dense,BatchNormalization, Dropout
from keras import optimizers, utils, regularizers
from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,TensorBoard

from keras.optimizers import SGD

#from sklearn.datasets import make_regression
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler

from time import localtime, strftime


#X1 = pd.read_csv('seq_data_a.csv',index_col=0)
#X2 = pd.read_csv('seq_data_b.csv',index_col=0)

#X = np.dstack((np.array(X1), np.array(X2)))

X = pd.read_csv('seq_data_cogen14.csv',index_col=0)[['seq01', 'seq02', 'seq03', 'seq04', 'seq05', 'seq06', 'seq07']]
Y = pd.read_csv('seq_data_cogen14.csv',index_col=0)[['seq08', 'seq09', 'seq10', 'seq11', 'seq12', 'seq13', 'seq14']]

out = np.random.permutation(len(X))
X_shuf = X.iloc[out]
Y_shuf = Y.iloc[out]

splt = int(np.floor(len(X_shuf)*0.90))

X_train = X_shuf[:splt]
Y_train = Y_shuf[:splt]

#np.save('X_train.npy',X_train)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_train = np.expand_dims(X_train, axis=2)
Y_train = np.expand_dims(Y_train, axis=2)

X_test = X_shuf[splt:]
Y_test = Y_shuf[splt:]

#np.save('X_test.npy',X_test)

X_test = np.array(X_test)
Y_test = np.array(Y_test)

X_test = np.expand_dims(X_test, axis=2)
Y_test = np.expand_dims(Y_test, axis=2)

model=load_model('auto_seq_model_dense7.h5')

adam = optimizers.Adam(lr=0.0001)

model.compile(optimizer=adam, loss='mse')

history = model.fit(X_train, Y_train,
                    epochs=300,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(X_test, Y_test),
                    verbose=1,
                    callbacks=[ReduceLROnPlateau(patience=30)])

model.save('auto_seq_model_cogen_forecast_trained.h5')

history = pd.DataFrame(history.history)

filename = 'history_'+strftime("%Y%m%d%H%M%S", localtime())
#filename = 'history_20200421164011.csv'

with open('history/'+filename+'.csv', 'a') as f:
    history = history.loc[pd.notnull(history.loss)] #remove NaN
    history.to_csv(f, header=True)

#plot loss

import matplotlib.pyplot as plt
df = history.loc[pd.notnull(history.loss)].iloc[:] #remove NaN

N=len(df["loss"])
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), df["loss"], label="train_loss")
plt.plot(np.arange(0, N), df["val_loss"], label="val_loss")
plt.legend(loc="upper right")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.savefig('history/'+filename+'.png')
plt.show()