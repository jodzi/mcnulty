# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:06:26 2015

@author: josephdziados
"""

import flask 
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import pickle

# Import model and scaler from pickle file
with open('logistic_model', 'r') as model:
    model = pickle.load(model)
with open('std_scale', 'r') as scaler:
    scaler = pickle.load(scaler)

# Initialize the app
app = flask.Flask(__name__)

# Homepage
@app.route('/')
def create_page():
    """
    Serves up my homepage of d3mnculty.html
    """
    with open('d3mcnulty2.html', 'r') as home:
        return home.read()
        
# Get an example and return its score
@app.route('/score', methods=["POST"])
def score():
    """
    When a POST request is sent made to this uri, read the example from the json,
    make a prediction, and send it with a response
    """
    # Get probability from our data
    data = flask.request.json
    x = np.matrix(data["example"])
    x_add = scaler.transform(x[0, (0,4,5,6,7,8)])
    x_scaled = np.delete(x, [0,4,5,6,7,8], axis=1)
    x_scaled = np.insert(x_scaled, (0,3,3,3,3,3), x_add, axis=1)
    prob = model.predict_proba(x_scaled)
    # Put the results in a dict to send as json
    results = {"prob": prob[0,1]}
    return flask.jsonify(results)
    
# Run server on port 80 (default web port)
app.run('0.0.0.0', debug=True, port=80)
