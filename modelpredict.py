import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from time import localtime, strftime

#X1 = pd.read_csv('seq_data_a.csv',index_col=0)
#X2 = pd.read_csv('seq_data_b.csv',index_col=0)

#X = np.dstack((np.array(X1), np.array(X2)))

data = pd.read_csv('seq_data_glass.csv',index_col=0)

#X = np.array(data[['seq01', 'seq02', 'seq03', 'seq04', 'seq05', 'seq06', 'seq07', 'seq08', 'seq09',
#                                                'seq10','seq11', 'seq12', 'seq13', 'seq14', 'seq15', 'seq16', 'seq17',
#                                                'seq18', 'seq19','seq20','seq21', 'seq22', 'seq23', 'seq24', 'seq25',
#                                                'seq26', 'seq27', 'seq28']])

X = np.array(data[['seq01', 'seq02', 'seq03', 'seq04', 'seq05', 'seq06', 'seq07']])

X = np.expand_dims(X, axis=2)

model=load_model('auto_seq_model_glass_trained.h5')
X_pred = model.predict(X)

#get the error term
mse = np.mean(np.power(X - X_pred, 2), axis=1)
#now add them to our data frame
data['MSE'] = mse

np.save('mse_glass.npy',mse)

#get percentile of each error
percentile_arr = np.zeros(len(data))
for i in range(len(data)):
    percentile_arr[i] = sum(data["MSE"] < data.iloc[i]['MSE'] )/len(data)
#add percentile to data frame
data['percentile']=percentile_arr

#find error threshold for outlier at 99th percentile
mse_threshold = np.quantile(data['MSE'], 0.99)
print(f'MSE 0.99 threshhold:{mse_threshold}')
#add outlier to dataframe
data['MSE_Outlier'] = 0
data.loc[data['MSE'] > mse_threshold, 'MSE_Outlier'] = 1

#save prediction result
filename = 'predicted_glass'+strftime("%Y%m%d%H%M%S", localtime())

data.to_csv('predicted/'+filename+'.csv', header=True)