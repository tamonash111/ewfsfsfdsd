import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app=application


## import ridge regresor model and standard scaler pickle
ridge_model=pickle.load(open('models/ridge-2.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler-2.pkl','rb'))

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        input_year = int(request.form.get('year'))
        km_driven = int(request.form.get('km_driven'))
        fuel =  int( request.form.get('fuel'))
        transmission = int(request.form.get('transmission'))
        owner =int(request.form.get('owner'))
        mileage = float(request.form.get('mileage'))
        engine = int(request.form.get('engine'))
        max_power = float(request.form.get('max_power'))
        seats = float(request.form.get('seats'))

        # Assuming standard_scaler is defined and ridge_model is loaded.
        new_data_scaled = standard_scaler.transform([[input_year, km_driven,fuel,transmission,owner, mileage, engine, max_power, seats]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html', result=result[0])
    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")
