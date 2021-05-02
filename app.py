# importing necessary libraries and functions
import numpy as np
import pandas as pd
import os
from flask import Flask, flash, request, jsonify, render_template, make_response, redirect
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import pickle
import io
from io import StringIO
import csv
import uuid
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__) #Initialize the flask App
app.secret_key = "abc"  

UPLOAD_FOLDER = "/Volumes/Transcend/IIT LEVEL 6/prototype-fraud/Application"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'csv'}

def only_allowed_files(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

      if f.filename == '':
        flash('No file is detected. Please enter a CSV file!!')
        return redirect('/')  

      if not only_allowed_files(f.filename):
        flash('Only CSV files are accepted!!') 
        return redirect('/')  

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input_file = csv.reader(stream)

    for row in csv_input_file:
        print(row)

    stream.seek(0)
    result = stream.read()

    # add file to dataframe
    df = pd.read_csv(StringIO(result))
    # df = pd.read_csv('newcleaned_test.csv')
    id_value = df['click_id']
    column = df.drop(columns=['click_id','click_time'])
    # print (df)
    # print (column)
    # print (id_value)

    # load the model to the system
    load_model = pickle.load(open('model.pkl', 'rb'))
    #prediction start
    prediction = load_model.predict(column)

    # print (prediction)
    #save csv
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