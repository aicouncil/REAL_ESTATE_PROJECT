import streamlit as st
import joblib
import json
import numpy as np

#Artifacts
_model = joblib.load('real_estate.pkl')
_scaling = joblib.load('feature_scaling.pkl')

my_json_file = open("training_data_columns.json" , "r")
my_json_file = json.load(my_json_file)
_features = my_json_file['data_columns']

#Extract unique location from the features for creating dropdown
location_info = [value for value in _features if value.startswith('location_')]

#to convert user input for area_type & availability into numeric form
area_type_function = lambda area_type : 0 if area_type.lower().strip() == 'super built-up area' else 1
availability_function = lambda availability : 0 if availability == 'Ready To Move' else 1

#---- streamlit UI-----
st.set_page_config(page_title='Your Real Estate Price Predictor' , layout='centered')

st.title('House Price Predictor')
st.markdown("Enter the property details to get an estimated price.....")

#Input fields
col1 , col2 = st.columns(2)
with col1:
    area_type = st.selectbox("Area Type" , ("super built-up area" , "Others"))
    availability = st.selectbox("Availability" , ("Ready To Move" , "UC"))
    size = st.number_input("Number of Bedrooms (BHK)" , min_value=1, max_value=6)


with col2:
    total_sqft = st.number_input("Total Square feet (e.g. 1200)" , min_value=100, max_value=4965)
    bath = st.number_input("Number of Bathrooms" , min_value=0, max_value=8)
    balcony = st.number_input("Number of Balconies" , min_value=0, max_value=3)


location = st.selectbox("Location" , location_info)

if st.button("Predict Price"):

    input = np.zeros(len(_features))

    input[0] = area_type_function(area_type)
    input[1] = availability_function(availability)
    input[2] = size
    input[3] = total_sqft
    input[4] = bath
    input[5] = balcony

    input[_features.index(location)] = 1

    price_predicted = np.exp(_model.predict(_scaling.transform([input])))[0]

    st.success(f"Estimated price: {price_predicted:,.2f} Lakhs")
    st.info("Note: This is an estimated price based on our model and avaulabel data, Actual price may vary a bit")