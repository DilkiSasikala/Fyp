# importing necessary libraries and functions
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, make_response
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import pickle
import io
from io import StringIO
import csv
import uuid

app = Flask(__name__) #Initialize the flask App

@app.route('/') # Homepage
def home():
    return render_template('index.html')

@app.route('/success') # Homepage
def success():
    return render_template('success.html')


@app.route('/predict',methods=['GET', 'POST'])
def predict():

    # retrieving values from form
    if request.method == 'POST':
      f = request.files['data_file']
      if not f:
        return "No file"

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)

    for row in csv_input:
        print(row)

    stream.seek(0)
    result = stream.read()

    df = pd.read_csv(StringIO(result))
    # df = pd.read_csv('newcleaned_test.csv')
    id_value = df['click_id']
    column = df.drop(columns=['click_id','click_time'])
    # print (df)
    # print (column)
    # print (id_value)

    # load the model from disk
    loaded_model = pickle.load(open('model.pkl', 'rb'))
    prediction = loaded_model.predict(column)

    # print (prediction)
    unique_filename = str(uuid.uuid4())
    res = pd.DataFrame(prediction)
    res.index = id_value
    res.columns = ["prediction"]
    res.to_csv(unique_filename+".csv")

    return render_template('success.html')
    # return 'prediction'


@app.route('/',methods=['GET', 'POST'])
def back():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)