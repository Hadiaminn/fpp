import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

# Dictionary to map displayed model names to encoded values
model_mapping = {
    'B-MAX': 0, 'C-MAX': 1, 'EcoSport': 2, 'Edge': 3, 'Escort': 4, 'Fiesta': 5,
    'Focus': 6, 'Fusion': 7, 'Galaxy': 8, 'Grand C-MAX': 9, 'Grand Tourneo Connect': 10,
    'KA': 11, 'Ka+': 12, 'Kuga': 13, 'Mondeo': 14, 'Mustang': 15, 'Puma': 16,
    'Ranger': 17, 'S-MAX': 18, 'Streetka': 19, 'Tourneo Connect': 20, 'Tourneo Custom': 21,
    'Transit Tourneo': 22
}

# Dictionary to map displayed year values to encoded values
year_mapping = {
    '1996': 0, '1998': 1, '2000': 2, '2002': 3, '2003': 4, '2004': 5,
    '2005': 6, '2006': 7, '2007': 8, '2008': 9, '2009': 10, '2010': 11,
    '2011': 12, '2012': 13, '2013': 14, '2014': 15, '2015': 16, '2016': 17,
    '2017': 18, '2018': 19, '2019': 20, '2020': 21
}

# Dictionary to map displayed transmission names to encoded values
transmission_mapping = {
    'Automatic':0, 'Manual':1, 'Semi-Automatic':2}

# Dictionary to map displayed fuel names to encoded values
fuelType_mapping = {
   'Diesel':0, 'Electric':1, 'Hybrid':2, 'Other':3, 'Petrol':4}

st.write("""
# Ford Car Price Prediction App

This app predicts the **Price** for a Ford car based on user input.
""")

st.sidebar.header('Select car features')

def user_input_features():
    st.write('### Select car features:')
    
    model = st.sidebar.selectbox('Car Model', list(model_mapping.keys()))
    year = st.sidebar.selectbox('Year of Registration', list(year_mapping.keys()))
    transmission = st.sidebar.selectbox('Transmission Type',list(transmission_mapping.keys()))
    mileage = st.sidebar.slider('Mileage (in miles)', 1.0, 177644.0, 15.0)
    fuelType = st.sidebar.selectbox('Fuel Type', list(fuelType_mapping.keys()))
    engineSize = st.sidebar.slider('Engine Size (in liters)', 0.0, 5.0, 1.0)

    data = {
        'model': model_mapping[model],
        'year': year_mapping[year],
        'transmission': transmission_mapping[transmission],
        'mileage': mileage,
        'fuelType': fuelType_mapping[fuelType],
        'engineSize': engineSize
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

#st.subheader('User Input Parameters')
st.write(df)

# Scale the input features using MinMaxScaler
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(pd.DataFrame(df))
scaled_features1 = pd.DataFrame(scaled_features, index=[0])
st.write(scaled_features1)

# Load the model
loaded_model = pickle.load(open("ford_price_prediction_model.h5", "rb"))

# Make prediction
prediction = loaded_model.predict(scaled_features1)

st.subheader('Predicted Car Price')
st.write('RM', round(prediction[0], 2))
