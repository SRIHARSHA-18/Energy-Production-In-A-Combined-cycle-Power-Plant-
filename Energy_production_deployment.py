# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:17:08 2023

@author: srinivasa
"""

import pandas as pd
import streamlit as st 
#from sklearn.model_selection import GridSearchCV
#from sklearn import preprocessing

st.title('Combined-Cycle Power Plant Energy Prediction App')

html_temp = """ 
<div style ="background-color:yellow;padding:13px"> 
<h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> 
</div> 
"""
st.write("This app predicts net hourly electrical energy output.")

st.sidebar.header('User Input Parameters')

def user_input_features():
    temperature = st.sidebar.number_input("Temperature (°C)")
    exhaust_vacuum = st.sidebar.number_input("Exhaust Vacuum (cm Hg)")
    amb_pressure = st.sidebar.number_input("Ambient Pressure (millibar)")
    r_humidity = st.sidebar.number_input("Relative Humidity (%)")
    data = {'temperature': [temperature],
            'exhaust_vacuum': [exhaust_vacuum],
            'amb_pressure': [amb_pressure],
            'r_humidity': [r_humidity]}
    
    features = pd.DataFrame(data,index = [0])
    return features 

df = user_input_features()
st.subheader('User Input parameters')
st.write(df)
 
# Read the dataset 
ccpp = pd.read_csv("C:/Users/Srinivasa/OneDrive/Desktop/prowork/project/P285/CCPP.csv")

# Remove duplicates 
ccpp_new = ccpp.drop_duplicates().reset_index(drop=True)

# Remove outliers
ccpp_new.drop(ccpp_new.index[ccpp_new['amb_pressure']>1029], inplace=True)
ccpp_new.drop(ccpp_new.index[ccpp_new['amb_pressure']<997], inplace=True)
ccpp_new.drop(ccpp_new.index[ccpp_new['r_humidity']<31], inplace=True)

# Define the target variables and features
#from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Define your target variable and features

X = ccpp_new.drop('energy_production', axis=1)
y = ccpp_new['energy_production']

# Gradient Boosting Regression
gbr_model = GradientBoostingRegressor(learning_rate= 0.1, max_depth=5, n_estimators= 300, random_state=42)

# Train the best model
gbr_model.fit(X, y)

# Make predictions
prediction = gbr_model.predict(df)

# Display the prediction
st.subheader('Predicted Result')
st.subheader('GBR')
if st.button("Predict"):
    st.markdown(f"<span style='font-size: 24px;'>Predicted Net Hourly Electrical Energy Output: {prediction[0]:.2f} MW</span>", unsafe_allow_html=True)

st.write("Hope you got your correct prediction Thank you!")





