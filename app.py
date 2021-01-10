
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
import json
import pickle
import pandas as pd

app = Flask(__name__)

knn_model = pickle.load(open('knn.sav', 'rb'))
ada_model = pickle.load(open('ada.sav', 'rb'))

@app.route('/knn', methods=['POST'])
def index():
    apply_model(request, 'knn')
    
@app.route('/ada', methods=['POST'])
def index():
    apply_model(request, 'ada')
    


def apply_model(req, model):
    # Store JSON and parse it
    json = req.json
    df = pd.read_json(json)

    ml_df = df
    ml_df.drop(labels=['weekdays', 'data_channel'], axis = 1, inplace=True)

    # Apply model to the data
    if model == "knn":
        prediction = knn_model.predict(ml_df)
    elif model == "ada":
        prediction = ada_model.predict(ml_df)

    # Return answer
    return json.dumps(list(prediction)), 200, {'ContentType':'application/json'}



if __name__ == '__main__':
    app.run('127.0.01', 8080, True)