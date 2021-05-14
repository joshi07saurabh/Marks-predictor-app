import numpy as np
import pandas as pd
from flask import Flask,request,render_template
import joblib
import pickle
app=Flask(__name__)

model=joblib.load("student_mark_predictor_model.pkl")



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    if request.method=='POST':

        input_features=[int(x) for x in request.form.values()]
        features_value=np.array(input_features)

        if input_features[0] <=0 or input_features[0] >24:
            return render_template('index.html',prediction_text='Please enter valid hours:')
        
        result=model.predict([features_value])[0][0].round(2)
        if result > 100:
            return render_template('index.html',prediction_text="Excellent You will get 100%")
        return render_template('index.html',prediction_text="You will get [{}%]  marks, when you do study [{}] hours per day.".format(result,int(features_value[0])))

if __name__=='__main__':
    app.run(debug=True)