# import the necessary packages
#from keras.models import load_model
from tensorflow.keras.models import load_model
import numpy as np
import flask
import pyodbc

import datetime as DT
import pandas as pd
from statistics import mean
# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

def load_models():
    # load the pre-trained Keras model

    model_glass = load_model('auto_seq_model_glass_trained.h5', compile=False)
    model_cogen = load_model('auto_seq_model_cogen_trained.h5', compile=False)
    models = {'glass':model_glass,'cogen':model_cogen}
    # load mean square error for calculating percentile

    mse_glass = np.load('mse_glass.npy')
    mse_cogen = np.load('mse_cogen.npy')
    mse_arrays = {'glass':mse_glass,'cogen':mse_cogen}

    return models, mse_arrays

def query_db():
    today = DT.date.today()
    week_ago = today - DT.timedelta(days=8)

    today_s = DT.date.today().strftime("%Y-%m-%d")
    week_ago_s= week_ago.strftime("%Y-%m-%d")

    print(today_s)
    print(week_ago_s)

    cnxn = pyodbc.connect('Driver={SQL Server};'
    #cnxn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
                          'Server=ptt-db-p05.ptt.corp;'
                          'Database=PTT-NGRAnywhere;'
                          'UID=ngranywhererusr;'
                          'PWD=cngranywhererusr;'
                          'Trusted_Connection=no;')

    query = "SELECT [Date], [SoldTo],[ShipTo], [ShipToName],[ContractType], [Description],[Volume], [DemandPlan],[MDCQ], [BillingType] FROM dbo.V_DataGSMGasdaily WHERE  [DATE] BETWEEN '%s' AND  '%s' ORDER BY [DATE] ;"  %(week_ago_s, today_s)
    df = pd.read_sql(query, cnxn)
    return df

factories = {'glass':['0030000001','0030000009','0030000083','0030000162','0030043703','0030000137','0030011637',
                      '0030019567','0030044623','0030000054','0030000136','0030030514','0030059837','0030000010',
                      '0030021634','0030000025','0030015470','0030055031'],
             'cogen':['0030000049','0030000071','0030015621','0030016267','0030000022','0030000123','0030015236',
                      '0030020016','0030033172','0030064789','0030000062','0030000072','0030000094','0030013680',
                      '0030014300','0030020778','0030030499','0030030778','0030040549','0030047048','0030062578',
                      '0030063296','0030008767','0030020317','0030020406','0030025358','0030000027','0030022988',
                      '0030024290','0030026631','0030050402','0030053591']}

models, mse_arrays = load_models()

@app.route("/result_auto", methods=["GET"]) # return result from auto encoder algorithm
def result_auto():
    data = {"success": False}
    # arrange sequence data
    df = query_db()
    df_seq_glass = pd.DataFrame(columns=['Date', 'ShipTo', 'seq01', 'seq02', 'seq03', 'seq04', 'seq05', 'seq06', 'seq07'])
    for factory_code in factories['glass']:
        #factory_code = '00' + str(factory_code)
        df_factory = df.loc[df['ShipTo'] == factory_code]
        df_factory = df_factory.drop_duplicates(subset=['Date'])
        #df_factory = df_factory.loc[df_factory['Description'] == 'Industrial-CoGen']  # apply only cogen factory
        if len(df_factory) > 0:
            df_seq = df_factory['Volume'].iloc[-7:]
            df_seq = np.expand_dims(df_seq, axis=0)
            df_seq = pd.DataFrame(df_seq, columns=['seq01', 'seq02', 'seq03', 'seq04', 'seq05', 'seq06', 'seq07'])
            df_seq['Date'] = df_factory.iloc[-1]['Date']
            df_seq['ShipTo'] = factory_code
            df_seq_glass = pd.concat([df_seq_glass,df_seq])
    df_seq_cogen = pd.DataFrame(
        columns=['Date', 'ShipTo', 'seq01', 'seq02', 'seq03', 'seq04', 'seq05', 'seq06', 'seq07'])
    for factory_code in factories['cogen']:
        #factory_code = '00' + str(factory_code)
        df_factory = df.loc[df['ShipTo'] == factory_code]
        df_factory = df_factory.loc[df_factory['Description'] == 'Industrial-CoGen']  # apply only cogen factory
        df_factory = df_factory.drop_duplicates(subset=['Date'])
        if len(df_factory) > 0:
            df_seq1 = df_factory['Volume'].iloc[-7:]
            df_seq1 = np.expand_dims(df_seq1, axis=0)
            df_seq1 = pd.DataFrame(df_seq1, columns=['seq01', 'seq02', 'seq03', 'seq04', 'seq05', 'seq06', 'seq07'])
            df_seq1['Date'] = df_factory.iloc[-1]['Date']
            df_seq1['ShipTo'] = factory_code
            df_seq_cogen = pd.concat([df_seq_cogen,df_seq1])

    #predict mse and percentile
    X = np.array(df_seq_glass[['seq01', 'seq02', 'seq03', 'seq04', 'seq05', 'seq06', 'seq07']])
    X = np.expand_dims(X, axis=2)
    X_pred = models['glass'].predict(X)

    # find mean square erorr between input and reconstructed
    mse = np.mean(np.power(X - X_pred, 2), axis=1)
    # now add them to our data frame
    df_seq_glass['MSE'] = mse

    # get percentile of each error
    percentile_arr = np.zeros(len(df_seq_glass))
    for i in range(len(df_seq_glass)):
        percentile_arr[i] = sum(mse_arrays['glass'] < mse[i]) / len(mse_arrays['glass'])
    # add percentile to data frame
    df_seq_glass['percentile'] = percentile_arr
    df_seq_glass = df_seq_glass.set_index('ShipTo')

    # predict mse and percentile
    X = np.array(df_seq_cogen[['seq01', 'seq02', 'seq03', 'seq04', 'seq05', 'seq06', 'seq07']])
    X = np.expand_dims(X, axis=2)
    X_pred = models['cogen'].predict(X)

    # find mean square erorr between input and reconstructed
    mse = np.mean(np.power(X - X_pred, 2), axis=1)
    # now add them to our data frame
    df_seq_cogen['MSE'] = mse

    # get percentile of each error
    percentile_arr = np.zeros(len(df_seq_cogen))
    for i in range(len(df_seq_cogen)):
        percentile_arr[i] = sum(mse_arrays['cogen'] < mse[i]) / len(mse_arrays['cogen'])
    # add percentile to data frame
    df_seq_cogen['percentile'] = percentile_arr
    df_seq_cogen = df_seq_cogen.set_index('ShipTo')

    # indicate that the request was a success
    data["success"] = True

    # construct output dict
    data["glass"] = df_seq_glass[['Date','percentile']].to_dict(orient='index')
    data["cogen"] = df_seq_cogen[['Date','percentile']].to_dict(orient='index')

    return flask.jsonify(data)

@app.route("/result_rule", methods=["GET"]) # return result from rule base algorithm
def result_rule():
    data = {"success": False}
    # arrange sequence data
    df = query_db()
    df_filtered = df[df['BillingType'].isin(['AMR', 'ADQ, AMR', 'AMR, FlowComp', 'FlowComp'])]
    df_filtered = df_filtered[df_filtered['ContractType'] == '151']
    df_glass_cogen = df_filtered[df_filtered['ShipTo'].isin(factories['glass'] + factories['cogen'])]
    glass_cogen_list = list(df_glass_cogen['ShipTo'].drop_duplicates())
    df_general = df_filtered[~df_filtered['ShipTo'].isin(factories['glass'] + factories['cogen'])]
    df_general['ShipToName'] = df_general['ShipToName'].fillna('บริษัท')
    general_list = list(df_general['ShipTo'].drop_duplicates())

    # create output for general factory
    df_rule_general = pd.DataFrame(
        columns=['Date', 'ShipTo', 'rule01', 'rule02', 'rule03', 'rule04', 'rule05', 'rule06', 'rule07'])
    for factory in general_list:
        df_factory = df_general[df_general['ShipTo'] == factory]
        if len(df_factory) > 3:
            df_rule = pd.DataFrame({'Date': df_factory.iloc[-1]['Date'], 'ShipTo': factory}, index=[0])
            df_rule['rule01'] = False
            df_rule['rule02'] = False
            df_rule['rule03'] = False
            df_rule['rule04'] = False
            df_rule['rule05'] = False
            df_rule['rule06'] = False
            df_rule['rule07'] = False
            #if df_factory['Volume'].iloc[-1] <= 0 and df_factory['Volume'].iloc[-2] > 0:
            if df_factory['Volume'].iloc[-1] <= 0 and df_factory['Volume'].iloc[-2] > 0:
                df_rule['rule01'] = True
                df_rule_general = pd.concat([df_rule_general, df_rule])
                continue
            if (df_factory['Volume'].iloc[-1] - df_factory['Volume'].iloc[-2]) / df_factory['Volume'].iloc[-2] > 0.9:
                df_rule['rule02'] = True
                df_rule_general = pd.concat([df_rule_general, df_rule])
                continue
            if (df_factory['Volume'].iloc[-1] - df_factory['Volume'].iloc[-2]) / df_factory['Volume'].iloc[-2] < -0.9:
                df_rule['rule03'] = True
                df_rule_general = pd.concat([df_rule_general, df_rule])
                continue
            if ((df_factory['Volume'].iloc[-2] - df_factory['Volume'].iloc[-3]) / df_factory['Volume'].iloc[
                -3] > 0.5 and
                    (df_factory['Volume'].iloc[-1] - df_factory['Volume'].iloc[-3]) / df_factory['Volume'].iloc[
                        -3] > 0.5):
                df_rule['rule04'] = True
                df_rule_general = pd.concat([df_rule_general, df_rule])
                continue
            if ((df_factory['Volume'].iloc[-2] - df_factory['Volume'].iloc[-3]) / df_factory['Volume'].iloc[
                -3] < -0.5 and
                    (df_factory['Volume'].iloc[-1] - df_factory['Volume'].iloc[-3]) / df_factory['Volume'].iloc[
                        -3] < -0.5):
                df_rule['rule05'] = True
                df_rule_general = pd.concat([df_rule_general, df_rule])
                continue
            if (df_factory['Volume'].iloc[-1] - mean(df_factory['Volume'])) / mean(df_factory['Volume']) > 0.3:
                df_rule['rule06'] = True
                df_rule_general = pd.concat([df_rule_general, df_rule])
                continue
            if (df_factory['Volume'].iloc[-1] - mean(df_factory['Volume'])) / mean(df_factory['Volume']) < -0.3:
                df_rule['rule07'] = True
            df_rule_general = pd.concat([df_rule_general, df_rule])
    df_rule_general = df_rule_general.set_index('ShipTo')

    # create output for glass and cogen factory
    df_rule_glass_cogen = pd.DataFrame(
        columns=['Date', 'ShipTo', 'rule01', 'rule02', 'rule03', 'rule04', 'rule05', 'rule06', 'rule07'])
    for factory in glass_cogen_list:
        df_factory = df_glass_cogen[df_glass_cogen['ShipTo'] == factory]
        if len(df_factory) > 3:
            df_rule = pd.DataFrame({'Date': df_factory.iloc[-1]['Date'], 'ShipTo': factory}, index=[0])
            df_rule['rule01'] = False
            df_rule['rule02'] = False
            df_rule['rule03'] = False
            df_rule['rule04'] = False
            df_rule['rule05'] = False
            df_rule['rule06'] = False
            df_rule['rule07'] = False
            # if df_factory['Volume'].iloc[-1] <= 0 and df_factory['Volume'].iloc[-2] > 0:
            if df_factory['Volume'].iloc[-1] <= 0 and df_factory['Volume'].iloc[-2] > 0:
                df_rule['rule01'] = True
                df_rule_glass_cogen = pd.concat([df_rule_glass_cogen, df_rule])
                continue
            if (df_factory['Volume'].iloc[-1] - df_factory['Volume'].iloc[-2]) / df_factory['Volume'].iloc[-2] > 0.9:
                df_rule['rule02'] = True
                df_rule_glass_cogen = pd.concat([df_rule_glass_cogen, df_rule])
                continue
            if (df_factory['Volume'].iloc[-1] - df_factory['Volume'].iloc[-2]) / df_factory['Volume'].iloc[-2] < -0.9:
                df_rule['rule03'] = True
                df_rule_glass_cogen = pd.concat([df_rule_glass_cogen, df_rule])
                continue
            if ((df_factory['Volume'].iloc[-2] - df_factory['Volume'].iloc[-3]) / df_factory['Volume'].iloc[
                -3] > 0.5 and
                    (df_factory['Volume'].iloc[-1] - df_factory['Volume'].iloc[-3]) / df_factory['Volume'].iloc[
                        -3] > 0.5):
                df_rule['rule04'] = True
                df_rule_glass_cogen = pd.concat([df_rule_glass_cogen, df_rule])
                continue
            if ((df_factory['Volume'].iloc[-2] - df_factory['Volume'].iloc[-3]) / df_factory['Volume'].iloc[
                -3] < -0.5 and
                    (df_factory['Volume'].iloc[-1] - df_factory['Volume'].iloc[-3]) / df_factory['Volume'].iloc[
                        -3] < -0.5):
                df_rule['rule05'] = True
                df_rule_glass_cogen = pd.concat([df_rule_glass_cogen, df_rule])
                continue
            if (df_factory['Volume'].iloc[-1] - mean(df_factory['Volume'])) / mean(df_factory['Volume']) > 0.3:
                df_rule['rule06'] = True
                df_rule_glass_cogen = pd.concat([df_rule_glass_cogen, df_rule])
                continue
            if (df_factory['Volume'].iloc[-1] - mean(df_factory['Volume'])) / mean(df_factory['Volume']) < -0.3:
                df_rule['rule07'] = True
            df_rule_glass_cogen = pd.concat([df_rule_glass_cogen, df_rule])
    df_rule_glass_cogen = df_rule_glass_cogen.set_index('ShipTo')

    # indicate that the request was a success
    data["success"] = True

    # construct output dict
    data["general"] = df_rule_general.to_dict(orient='index')
    data["glass_cogen"] = df_rule_glass_cogen.to_dict(orient='index')

    return flask.jsonify(data)

@app.route("/result_rule_test", methods=["GET"]) # return result from rule base algorithm
def result_rule_test():
    data = {"success": False}
    # arrange sequence data
    df = query_db()
    df_filtered = df[df['BillingType'].isin(['AMR', 'ADQ, AMR', 'AMR, FlowComp', 'FlowComp'])]
    df_filtered = df_filtered[df_filtered['ContractType'] == '151']
    df_glass_cogen = df_filtered[df_filtered['ShipTo'].isin(factories['glass'] + factories['cogen'])]
    glass_cogen_list = list(df_glass_cogen['ShipTo'].drop_duplicates())
    df_general = df_filtered[~df_filtered['ShipTo'].isin(factories['glass'] + factories['cogen'])]
    df_general['ShipToName'] = df_general['ShipToName'].fillna('บริษัท')
    general_list = list(df_general['ShipTo'].drop_duplicates())

    # create output for general factory
    df_rule_general = pd.DataFrame(
        columns=['Date', 'ShipTo', 'ShipToName', 'Type', 'rule01', 'rule02', 'rule03', 'rule04', 'rule05', 'rule06', 'rule07'])
    for factory in general_list:
        df_factory = df_general[df_general['ShipTo'] == factory]
        if len(df_factory) > 3:
            df_rule = pd.DataFrame({'Date': df_factory.iloc[-1]['Date'], 'ShipTo': factory, 'ShipToName': df_factory.iloc[-1]['ShipToName'], 'Type': 'general'}, index=[0])
            df_rule['rule01'] = False
            df_rule['rule02'] = False
            df_rule['rule03'] = False
            df_rule['rule04'] = False
            df_rule['rule05'] = False
            df_rule['rule06'] = False
            df_rule['rule07'] = False
            #if df_factory['Volume'].iloc[-1] <= 0 and df_factory['Volume'].iloc[-2] > 0:
            if df_factory['Volume'].iloc[-1] <= 0 and df_factory['Volume'].iloc[-2] > 0:
                df_rule['rule01'] = True
                df_rule_general = pd.concat([df_rule_general, df_rule])
                continue
            if (df_factory['Volume'].iloc[-1] - df_factory['Volume'].iloc[-2]) / df_factory['Volume'].iloc[-2] > 0.9:
                df_rule['rule02'] = True
                df_rule_general = pd.concat([df_rule_general, df_rule])
                continue
            if (df_factory['Volume'].iloc[-1] - df_factory['Volume'].iloc[-2]) / df_factory['Volume'].iloc[-2] < -0.9:
                df_rule['rule03'] = True
                df_rule_general = pd.concat([df_rule_general, df_rule])
                continue
            if ((df_factory['Volume'].iloc[-2] - df_factory['Volume'].iloc[-3]) / df_factory['Volume'].iloc[
                -3] > 0.5 and
                    (df_factory['Volume'].iloc[-1] - df_factory['Volume'].iloc[-3]) / df_factory['Volume'].iloc[
                        -3] > 0.5):
                df_rule['rule04'] = True
                df_rule_general = pd.concat([df_rule_general, df_rule])
                continue
            if ((df_factory['Volume'].iloc[-2] - df_factory['Volume'].iloc[-3]) / df_factory['Volume'].iloc[
                -3] < -0.5 and
                    (df_factory['Volume'].iloc[-1] - df_factory['Volume'].iloc[-3]) / df_factory['Volume'].iloc[
                        -3] < -0.5):
                df_rule['rule05'] = True
                df_rule_general = pd.concat([df_rule_general, df_rule])
                continue
            if (df_factory['Volume'].iloc[-1] - mean(df_factory['Volume'])) / mean(df_factory['Volume']) > 0.3:
                df_rule['rule06'] = True
                df_rule_general = pd.concat([df_rule_general, df_rule])
                continue
            if (df_factory['Volume'].iloc[-1] - mean(df_factory['Volume'])) / mean(df_factory['Volume']) < -0.3:
                df_rule['rule07'] = True
            df_rule_general = pd.concat([df_rule_general, df_rule])
    df_rule_general = df_rule_general.set_index('ShipTo')

    # create output for glass and cogen factory
    df_rule_glass_cogen = pd.DataFrame(
        columns=['Date', 'ShipTo', 'ShipToName', 'Type', 'rule01', 'rule02', 'rule03', 'rule04', 'rule05', 'rule06', 'rule07'])
    for factory in glass_cogen_list:
        df_factory = df_glass_cogen[df_glass_cogen['ShipTo'] == factory]
        if len(df_factory) > 3:
            df_rule = pd.DataFrame({'Date': df_factory.iloc[-1]['Date'], 'ShipTo': factory, 'ShipToName': df_factory.iloc[-1]['ShipToName'], 'Type': 'glass_cogen'}, index=[0])
            df_rule['rule01'] = False
            df_rule['rule02'] = False
            df_rule['rule03'] = False
            df_rule['rule04'] = False
            df_rule['rule05'] = False
            df_rule['rule06'] = False
            df_rule['rule07'] = False
            # if df_factory['Volume'].iloc[-1] <= 0 and df_factory['Volume'].iloc[-2] > 0:
            if df_factory['Volume'].iloc[-1] <= 0 and df_factory['Volume'].iloc[-2] > 0:
                df_rule['rule01'] = True
                df_rule_glass_cogen = pd.concat([df_rule_glass_cogen, df_rule])
                continue
            if (df_factory['Volume'].iloc[-1] - df_factory['Volume'].iloc[-2]) / df_factory['Volume'].iloc[-2] > 0.9:
                df_rule['rule02'] = True
                df_rule_glass_cogen = pd.concat([df_rule_glass_cogen, df_rule])
                continue
            if (df_factory['Volume'].iloc[-1] - df_factory['Volume'].iloc[-2]) / df_factory['Volume'].iloc[-2] < -0.9:
                df_rule['rule03'] = True
                df_rule_glass_cogen = pd.concat([df_rule_glass_cogen, df_rule])
                continue
            if ((df_factory['Volume'].iloc[-2] - df_factory['Volume'].iloc[-3]) / df_factory['Volume'].iloc[
                -3] > 0.5 and
                    (df_factory['Volume'].iloc[-1] - df_factory['Volume'].iloc[-3]) / df_factory['Volume'].iloc[
                        -3] > 0.5):
                df_rule['rule04'] = True
                df_rule_glass_cogen = pd.concat([df_rule_glass_cogen, df_rule])
                continue
            if ((df_factory['Volume'].iloc[-2] - df_factory['Volume'].iloc[-3]) / df_factory['Volume'].iloc[
                -3] < -0.5 and
                    (df_factory['Volume'].iloc[-1] - df_factory['Volume'].iloc[-3]) / df_factory['Volume'].iloc[
                        -3] < -0.5):
                df_rule['rule05'] = True
                df_rule_glass_cogen = pd.concat([df_rule_glass_cogen, df_rule])
                continue
            if (df_factory['Volume'].iloc[-1] - mean(df_factory['Volume'])) / mean(df_factory['Volume']) > 0.3:
                df_rule['rule06'] = True
                df_rule_glass_cogen = pd.concat([df_rule_glass_cogen, df_rule])
                continue
            if (df_factory['Volume'].iloc[-1] - mean(df_factory['Volume'])) / mean(df_factory['Volume']) < -0.3:
                df_rule['rule07'] = True
            df_rule_glass_cogen = pd.concat([df_rule_glass_cogen, df_rule])
    df_rule_glass_cogen = df_rule_glass_cogen.set_index('ShipTo')

    # construct output dict
    df_rule_combine = pd.concat([df_rule_general, df_rule_glass_cogen])
    combine_dict = df_rule_combine.to_dict(orient='index')
    data.update(combine_dict)
    # indicate that the request was a success
    data["success"] = True

    return flask.jsonify(data)

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the view
    data = {"success": False}
    model = 0
    # ensure an data was properly uploaded to our endpoint
    if flask.request.method == "POST":
        # read input dict

        X = flask.request.get_json(force=True)
        data['Date'] = X['Date']
        data['Ship to'] = X['Ship to']
        #select model and mse_array according to 'Ship to' code
        for type in factories:
            if int(X['Ship to']) in factories[type]:
                model = models[type]
                mse_array = mse_arrays[type]
                break
        if model == 0:
            print('=========invalid Ship to code, model not found========')
        # preprocess the data and prepare it for prediction
        X_pred = X['seq']
        X_pred = np.array(X_pred)
        X_pred = X_pred.reshape(1, -1)
        X_pred = np.expand_dims(X_pred, axis=2)

        # reconstruct input seq
        Y_pred = model.predict(X_pred)

        #find mean square erorr between input and reconstructed
        mse_pred = np.mean(np.power(Y_pred - X_pred, 2), axis=1)
        #find percentile of mse
        percentile = sum(mse_array < mse_pred) / len(mse_array)

        #put percentile in output data dict
        data["percentile"] = float(percentile)

        # indicate that the request was a success
        data["success"] = True
    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))

    #app.run(debug=True)
    app.run(host='0.0.0.0', port = 5000)
