import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join

main_path = 'data'

#create all dataframe
df = pd.DataFrame(columns=['ï»¿Date','Ship to','Description','Volume(MMSCFD)'])

for year_folder in listdir(main_path):
    year_path = main_path+'/'+year_folder
    for month_folder in listdir(join(year_path)):
        month_path = year_path+'/'+month_folder
        for file_name in listdir(month_path) :
            file_path = month_path+'/'+file_name
            if isfile(file_path):
                df = df.append(pd.read_csv(file_path, sep='|',engine='python')[['ï»¿Date','Ship to','Description','Volume(MMSCFD)']].iloc[:])
#sort df to decending date
df['ï»¿Date'] = pd.to_datetime(df['ï»¿Date'], format='%m-%d-%Y', errors='coerce')
df = df.sort_values(by='ï»¿Date', ascending=False)

print('df Done=============')

#create required sequence
#df_seq = pd.DataFrame(columns=['Date','Ship to','seq01','seq02','seq03','seq04','seq05','seq06','seq07','seq08','seq09','seq10',
#                               'seq11','seq12','seq13','seq14','seq15','seq16','seq17','seq18','seq19','seq20',
#                               'seq21','seq22','seq23','seq24','seq25','seq26','seq27','seq28'])

df_seq = pd.DataFrame(columns=['Date','Ship to','seq01','seq02','seq03','seq04','seq05','seq06','seq07'])

#require list of glass factory
#require_list = [30000001,30000009,30000083,30000162,30043703,30000137,30011637,30019567,30044623,30000054,30000136,
#                30030514,30059837,30000010,30021634,30000025,30015470,30055031]

#require list of cogen factory
require_list = [30000049,30000071,30015621,30016267,30000022,30000123,30015236,30020016,30033172,30064789,30000062,
                30000072,30000094,30013680,30014300,30020778,30030499,30030778,30040549,30047048,30062578,30063296,
                30008767,30020317,30020406,30025358,30000027,30022988,30024290,30026631,30050402,30053591]

for factory_code in require_list:
    df_factory = df.loc[df['Ship to'] == factory_code]
    df_factory = df_factory.loc[df_factory['Description'] == 'Industrial-CoGen'] #apply only cogen factory
    for i in range(len(df_factory)-7):
        df_seq1 = df_factory.iloc[i:i+7]['Volume(MMSCFD)']
        df_seq1 = np.expand_dims(df_seq1, axis=0)
        if False not in np.squeeze(df_seq1 >= 0):
            #df_seq1 = pd.DataFrame(df_seq1,columns=['seq01', 'seq02', 'seq03', 'seq04', 'seq05', 'seq06', 'seq07', 'seq08', 'seq09',
            #                                        'seq10','seq11', 'seq12', 'seq13', 'seq14', 'seq15', 'seq16', 'seq17',
            #                                        'seq18', 'seq19','seq20','seq21', 'seq22', 'seq23', 'seq24', 'seq25',
            #                                        'seq26', 'seq27', 'seq28'])

            df_seq1 = pd.DataFrame(df_seq1,columns=['seq01', 'seq02', 'seq03', 'seq04', 'seq05', 'seq06', 'seq07'])

            df_seq1['Date'] = df_factory.iloc[i]['ï»¿Date']
            df_seq1['Ship to'] = factory_code

            df_seq = df_seq.append(df_seq1, sort=True)
    print(factory_code, 'done')

df_seq.to_csv('seq_data_congen1.csv')