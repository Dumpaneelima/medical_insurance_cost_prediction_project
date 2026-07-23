import streamlit as st
import joblib
import pandas as pd
import numpy as np

model=joblib.load("insurance_model.pkl")
st.title("Medical Insurance Cost Prediction")
st.write("This application predicts the medical insurance cost based on the user input fields.")
#scaled
scaler=joblib.load("insurance_scaler.pkl")

### inputs
age=st.number_input("Age",min_value=1, max_value=100)

sex=st.selectbox(
    "Sex",
    ["male","female"]
)

bmi=st.number_input("BMI")

sex= 1 if sex == "male" else 0

children=st.number_input(
    "Children",
    min_value=0,
    max_value=10
)

smoker= st.selectbox(
    "Smoker",
    ["yes","no"]
)

region=st.selectbox(
    "Region",
    ["northeast","northwest","southeast","southwest"]
)

## covert categorical values
sex= 1 if sex == "male" else 0
smoker= 1 if smoker == "yes" else 0
region_mapping = {
    "northeast": 0,
    "northwest": 1,
    "southeast": 2,
    "southwest": 3
}

region = region_mapping[region]


####convert inputs according training preprocess
if st.button("Predict"):
    input_data=np.array(
        [[age,sex,bmi,children,smoker,region]]
    )


    #####apply predict
    prediction=model.predict(input_data)
    st.success(f"Predicted insurance charges: {prediction[0]:2f}")