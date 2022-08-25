# import the necessary packages
import requests
import numpy as np
import pandas as pd
import json
import time
from time import localtime, strftime
import traceback


# initialize the Keras REST API endpoint URL
KERAS_REST_API_URL = "http://172.19.3.173:5000/predict"
#KERAS_REST_API_URL = "http://app02-stg.inwini.com/rvp/predict"
#KERAS_REST_API_URL = "https://amr-anomaly-ce6ez7jgkq-uc.a.run.app/predict"


# load the input dict and construct the payload for the request

data = pd.read_csv('seq_data_glass.csv', index_col = 0).iloc[6015:]


for i in range(len(data)) :
    try:
        X = {'Date':data.iloc[i]['Date']}
        X['Ship to'] = str(data.iloc[i]['Ship to'])
        #X['Ship to'] = str(000000)
        #X['seq']= list(data.iloc[i][['seq01', 'seq02', 'seq03', 'seq04', 'seq05', 'seq06', 'seq07', 'seq08', 'seq09',
        #                                'seq10', 'seq11', 'seq12', 'seq13', 'seq14', 'seq15', 'seq16', 'seq17',
        #                                'seq18', 'seq19', 'seq20', 'seq21', 'seq22', 'seq23', 'seq24', 'seq25',
        #                                'seq26', 'seq27', 'seq28']])
        X['seq'] = list(data.iloc[i][['seq01', 'seq02', 'seq03', 'seq04', 'seq05', 'seq06', 'seq07']])
        payload = X
        start = time.time()
        # submit the request
        r = requests.post(KERAS_REST_API_URL, json=payload).json()
        # ensure the request was successful
        if r["success"]:
            # loop over the predictions and display them
            end = time.time()
            print("[INFO] calculation at {} for sample number {} took {:.6f} seconds".format(strftime("%Y%m%d%H%M%S", localtime()),i,end - start))
            print('Date =', r["Date"])
            print('Ship to =', r["Ship to"])
            print('Percentile = ', r["percentile"])
            print('JSON = ', r)
            print('==========================')

        # otherwise, the request failed
        else:
            print("Request failed")
        time.sleep(5)

    except Exception as e:
        print("[INFO] request fail at {} for sample number {}".format(strftime("%Y%m%d%H%M%S", localtime()), i))
        print('==========================')
        print(traceback.print_exc())
        time.sleep(5)

