
#import libraries
import numpy as np
import pandas as pd
import joblib
import xgboost

from flask import Flask, request, jsonify

#initialize flask
super_kart = Flask("Superkart")

#load the model
model = joblib.load("superkart_model.joblib")

#define needed data for prediction to work
features = ['Product_Weight',
            'Product_Sugar_Content',
            'Product_Allocated_Area',
            'Product_MRP',
            'Store_Size',
            'Store_Location_City_Type',
            'Store_Type',
            'Product_Id_char',
            'Product_Type_Category',
            'Store_Age_Years']


#api home endpoint
@super_kart.get("/")
def home():
    return("Welcome, the Super Kart system is ready")

#api single prediction
@super_kart.post("/v1/predict")
def predict_single():
    #get data from user
    data = request.get_json()

    #check if any data is missing
    missing = [f for f in features if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    #convert json into dataframe
    dfdata = pd.DataFrame([{f: data[f] for f in features}])

    prediction = model.predict(dfdata)

    return jsonify({"predictionjs": float(prediction[0])})

#api batch prediction
@super_kart.post("/v1/predictbatch")
def predict_batch():
    #get file from user
    batch_data = pd.read_csv(request.files["file"])

    #check if any data is missing
    missing_batch = [f for f in features if f not in batch_data.columns]
    if missing_batch:
        return jsonify({"error": f"Missing fields: {missing_batch}"}), 400

    #ensure consistent column order
    batch_data = batch_data[features]

    batch_prediction = model.predict(batch_data)

    return jsonify({str(i): round(float(p), 2) for i, p in enumerate(batch_prediction)})
