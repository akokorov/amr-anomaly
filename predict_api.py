# import the necessary packages
#from keras.models import load_model
from tensorflow.keras.models import load_model
import numpy as np
import flask
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

factories = {'glass':[30000001,30000009,30000083,30000162,30043703,30000137,30011637,30019567,30044623,30000054,30000136,
                30030514,30059837,30000010,30021634,30000025,30015470,30055031],
                'cogen':[30000049,30000071,30015621,30016267,30000022,30000123,30015236,30020016,30033172,30064789,30000062,
                30000072,30000094,30013680,30014300,30020778,30030499,30030778,30040549,30047048,30062578,30063296,
                30008767,30020317,30020406,30025358,30000027,30022988,30024290,30026631,30050402,30053591]}

models , mse_arrays = load_models()

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the view
    data = {"success": False}
    model = 0
    # ensure an data was properly uploaded to our endpoint
    if flask.request.method == "POST":
        # read input dict
        #param = flask.request.args.get('name')
        #form_val = flask.request.form.get('last')
        #file = flask.request.files.get('file')
        #file.save('data\input_test.csv')
        #print('from param ', param)
        #print('from form ', form_val)
        #print('from file ', file.filename)
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
