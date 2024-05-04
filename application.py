from flask import Flask, render_template, request
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Luxury watch clean.csv')

@app.route('/')
def index():
    return render_template('/Users/Rafael/Downloads/WatchEstimator/index.html')

if __name__=='__main__':
    app.run(debug=True)