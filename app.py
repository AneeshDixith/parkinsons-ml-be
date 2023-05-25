# -*- coding: utf-8 -*-
"""
Created on Mon May 22 20:34:49 2023

@author: AneeshDixit
"""

from flask import Flask
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
from keras.models import load_model
import numpy as np
import joblib

app = Flask(__name__)
api = Api(app)
CORS(app, resources={r"*": {"origins": "*"}})

req_args = reqparse.RequestParser()
req_args.add_argument("fo", type=float)
req_args.add_argument("flo", type=float)
req_args.add_argument("fhi", type=float)
req_args.add_argument("shimmer", type=float)
req_args.add_argument("shimmerDb", type=float)
req_args.add_argument("jitterPercent", type=float)
req_args.add_argument("jitterAbs", type=float)
req_args.add_argument("rap", type=float)
req_args.add_argument("ppq", type=float)
req_args.add_argument("ddp", type=float)
req_args.add_argument("apq3", type=float)
req_args.add_argument("apq5", type=float)
req_args.add_argument("apq", type=float)
req_args.add_argument("dda", type=float)
req_args.add_argument("nhr", type=float)
req_args.add_argument("hnr", type=float)
req_args.add_argument("rpde", type=float)
req_args.add_argument("d2", type=float)
req_args.add_argument("dfa", type=float)
req_args.add_argument("spread1", type=float)
req_args.add_argument("spread2", type=float)
req_args.add_argument("ppe", type=float)


class Home(Resource):
    def get(self):
        return {"data": "Hello There! Welcome to Parkinsons Classifier!"}


class ParkinsonsClassifier(Resource):
    def post(self):
        loaded_model = joblib.load('rf_model.sav')

        args = req_args.parse_args()

        x_one = np.array([args.get('fo'), args.get('fhi'),
                          args.get('flo'), args.get('jitterPercent'),
                          args.get('jitterAbs'), args.get(
                              'rap'), args.get('ppq'),
                          args.get('ddp'), args.get('shimmer'),
                          args.get('shimmerDb'), args.get('apq3'),
                          args.get('apq5'), args.get('apq'),
                          args.get('dda'), args.get('nhr'), args.get('hnr'),
                          args.get('rpde'), args.get(
                              'dfa'), args.get('spread1'),
                          args.get('spread2'), args.get('d2'), args.get('ppe')])

        float_features = np.array(x_one)
        float_features_one = float_features.reshape(1, -1)
        result = loaded_model.predict(float_features_one)
        print("Result: ", result)

        if result == 0:
            ty = "Negative"
        else:
            ty = "Positive"

        return {"result": ty}


api.add_resource(ParkinsonsClassifier, "/parkinsonsClassifier")
api.add_resource(Home, "/")

if __name__ == "__main__":
    app.run(debug=True)
