import pickle
from flask import Flask, request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

# import regressor and standard scaler pickel
regressor=pickle.load(open('models/1regressor.pkl', 'rb'))
standard_scaler=pickle.load(open('models/1scaler.pkl', 'rb'))


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predict", methods=["GET", "POST"])
def predict_data():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        WS=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        scaled_data=standard_scaler.transform([[Temperature,RH,WS,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=regressor.predict(scaled_data)

        return render_template('home.html', results=result[0])



    else:
        return render_template('home.html') 



if __name__=="__main__":
    app.run(host="0.0.0.0")
