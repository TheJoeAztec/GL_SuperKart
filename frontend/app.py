
#import libraries
import requests
import streamlit as st
import pandas as pd

backend_url = "http://superkart-backend:7860"

st.title("SuperKart Prediction System")


#data input
Product_Weight = st.number_input("Product Weight", min_value=0.0, value=5.0)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, value=0.07)
Product_MRP = st.number_input("Product MRP", min_value=0.0, value=147.0)
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Departmental Store", "Food Mart"])
Product_Id_char = st.selectbox("Product ID Character", ["FD", "DR", "NC"])
Store_Age_Years = st.number_input("Store Age in Years", min_value=0.0, value=24.0)
Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

#convert to json
predict_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category
}




if st.button("Predict", type="primary"):
    response = requests.post(f"{backend_url}/v1/predict", json=predict_data)
    if response.ok:
        result = response.json()
        prediction = result["predictionjs"]
        st.success(f"Predicted sales: {prediction:,.2f}")

    else:
        st.error(f"API error ({response.status_code}): {response.text}")

#Batch Prediction    
st.header("Batch Prediction")

uploaded = st.file_uploader("Upload a correctly formatted CSV", type="csv")
if uploaded is not None:
    if st.button("Batch Predict", type="primary"):
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        response = requests.post(
            f"{backend_url}/v1/predictbatch",
            files={"file": ("batch.csv", uploaded.getvalue(), "text/csv")},
            timeout=120,
        )

        if response.ok:
            #read results
            batch_prediction = pd.Series(response.json(), name="Predicted_Sales")
            batch_prediction.index = batch_prediction.index.astype(int)
            results = df.join(batch_prediction)

            #present results
            st.download_button("Download results", results.to_csv(index=False), "predictions.csv", "text/csv")
            st.dataframe(results)
        else:
            st.error(f"API error ({response.status_code}): {response.text}")

