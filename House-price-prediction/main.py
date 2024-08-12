import streamlit as st
import pickle
import json
import numpy as np

# Load the model and columns
model_path = 'banglore_home_prices_model.pickle'
columns_path = 'columns.json'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(columns_path, 'r') as f:
    data_columns = json.load(f)['data_columns']

def predict_price(location, sqft, bath, bhk):
    try:
        loc_index = data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return model.predict([x])[0]

# Streamlit app
st.title("Bangalore House Price Prediction")

# Input fields
location = st.selectbox("Location", sorted(data_columns[3:]))
sqft = st.number_input("Total Square Feet", min_value=300, max_value=10000, value=1000)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
bhk = st.number_input("Number of BHK", min_value=1, max_value=10, value=2)

if st.button("Predict"):
    prediction = predict_price(location, sqft, bath, bhk)
    st.write(f"The estimated price is {prediction:.2f} lakhs")

