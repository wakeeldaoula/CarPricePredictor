import pandas as pd
import numpy as np
import pickle
import streamlit as st

# Load dataset and model
dataset = pickle.load(open('Cleaned_Cars.pkl','rb'))
pipe = pickle.load(open('Model.pkl','rb'))

st.title("🚗 Car Price Predictor App")

# User inputs
company = st.selectbox("Select the company of the car", dataset['company'].unique())
name = st.selectbox("Select the name of the car", dataset['name'].unique())
fuel_type = st.selectbox("Select the fuel type of the car", dataset['fuel_type'].unique())
year = st.number_input("Enter the year of the car", min_value=1900, max_value=2025, step=1)
kms_driven = st.number_input("Enter the kilometers driven by the car")


# Predict button
if st.button("Predict"):

    # Create DataFrame in the same format as training data
    input_df = pd.DataFrame([[name, company, year, kms_driven, fuel_type]], 
                            columns=['name','company','year','kms_driven','fuel_type'])

    # Predict
    predicted_price = pipe.predict(input_df)



    st.success(f"The predicted price is: ₹ {int(predicted_price):,}")
