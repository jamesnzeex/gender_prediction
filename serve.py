import os
import train
import utils
import argparse
import numpy as np
import pandas as pd

from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--retrain", default=False, action='store_true', help="retrain model")
parser.add_argument("-d", "--deploy", default=False, action='store_true', help="deploy flask app")
parser.add_argument("-n", "--name", default=[], nargs='+', help="name(s) to be predicted")
args = parser.parse_args()
app = Flask(__name__)

def pred(name):
    try:
        df = pd.DataFrame({'name': name})
    except:
        name = [name]
        df = pd.DataFrame({'name': name}, index=[0])
    df = utils.preprocess(df, train=False)
    result = model.predict(np.asarray(df['name'].values.tolist())).squeeze(axis=1)
    df['gender'] = ['Male' if logit > 0.5 else 'Female' for logit in result]
    df['probability'] = [logit if logit > 0.5 else 1.0 - logit for logit in result]
    gender_prediction = dict(zip(name, df['gender'].to_list()))
    print (gender_prediction)
    return(gender_prediction)

@app.route('/', methods=['GET', 'POST'])
def predict():
    data = request.get_json()
    name = data.get('name')
    try:
        df = pd.DataFrame({'name': name})
    except:
        name = [name]
        df = pd.DataFrame({'name': name}, index=[0])
    df = utils.preprocess(df, train=False)
    result = model.predict(np.asarray(df['name'].values.tolist())).squeeze(axis=1)
    df['gender'] = ['Male' if logit > 0.5 else 'Female' for logit in result]
    df['probability'] = [logit if logit > 0.5 else 1.0 - logit for logit in result]
    gender_prediction = dict(zip(name, df['gender'].to_list()))
    return jsonify(gender_prediction)

if __name__ == '__main__':
    model_path = os.path.join(os.getcwd(), 'model/saved_model.h5')
    if args.retrain:
        model,_ = train.train(epoch=50)
    elif os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model,_ = train.train(epoch=50)
    if args.deploy:
        app.run()
    else:
        pred(args.name)