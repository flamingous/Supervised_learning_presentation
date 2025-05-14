import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib


# Set page configuration
st.set_page_config(page_title="Sustainable Fashion Predictor", layout="wide")

# Title and description
st.title("Sustainable Fashion Predictor")
st.markdown("""
This app predicts whether a fashion brand has **Recycling Programs** or uses **Eco-Friendly Manufacturing** 
based on brand details. Enter the information below and click 'Predict' to see the results.
""")

# Load dataset to get valid categories for dropdowns
@st.cache_data
def load_data():
    df = pd.read_csv('sustainable_fashion_trends_2024.csv')
    return df

df = load_data()

# Get unique values for dropdowns
countries = df['Country'].unique()
materials = df['Material_Type'].unique()
certifications = df['Certifications'].unique()
market_trends = df['Market_Trend'].unique()
sustainability_ratings = ['A', 'B', 'C', 'D']

# Input form
st.subheader("Enter Brand Details")
with st.form(key='prediction_form'):
    col1, col2 = st.columns(2)
    
    with col1:
        country = st.selectbox("Country", countries)
        year = st.slider("Year", min_value=2018, max_value=2030, value=2024)
        sustainability_rating = st.selectbox("Sustainability Rating", sustainability_ratings)
        material_type = st.selectbox("Material Type", materials)
        carbon_footprint = st.number_input("Carbon Footprint (Metric Tons)", min_value=0.0, value=10.0, step=0.1)
        
    with col2:
        water_usage = st.number_input("Water Usage (Liters)", min_value=0.0, value=500.0, step=10.0)
        waste_production = st.number_input("Waste Production (KG)", min_value=0.0, value=100.0, step=10.0)
        certification = st.selectbox("Certifications", certifications)
        avg_price = st.number_input("Average Price (USD)", min_value=0.0, value=50.0, step=1.0)
        product_lines = st.number_input("Number of Product Lines", min_value=1, max_value=50, value=5)
        market_trend = st.selectbox("Market Trend", market_trends)
    
    submit_button = st.form_submit_button(label="Predict")

# Load pre-trained models
@st.cache_resource
def load_models():
    rf_recycling = joblib.load('rf_recycling_model.joblib')
    rf_eco = joblib.load('rf_eco_friendly_model.joblib')
    return rf_recycling, rf_eco

rf_recycling, rf_eco = load_models()

# Function to preprocess input
def preprocess_input(data, feature_columns, categorical_columns):
    input_df = pd.DataFrame([data], columns=feature_columns)
    
    # Load or create label encoders based on dataset
    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        le.fit(df[col])  # Fit on original dataset
        encoders[col] = le
        input_df[col] = le.transform(input_df[col])
    
    return input_df, encoders

# Prediction function
def make_predictions(model, input_data, target_name):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0]
    return prediction[0], probability



# Process form submission
if submit_button:
    # Define features
    feature_columns = [
        'Country', 'Year', 'Sustainability_Rating', 'Material_Type', 'Carbon_Footprint_MT',
        'Water_Usage_Liters', 'Waste_Production_KG', 'Certifications', 'Average_Price_USD',
        'Product_Lines', 'Market_Trend'
    ]
    categorical_columns = [
        'Country', 'Sustainability_Rating', 'Material_Type', 'Certifications', 'Market_Trend'
    ]
    
    # Create input dictionary
    input_data = {
        'Country': country,
        'Year': year,
        'Sustainability_Rating': sustainability_rating,
        'Material_Type': material_type,
        'Carbon_Footprint_MT': carbon_footprint,
        'Water_Usage_Liters': water_usage,
        'Waste_Production_KG': waste_production,
        'Certifications': certification,
        'Average_Price_USD': avg_price,
        'Product_Lines': product_lines,
        'Market_Trend': market_trend
    }
    
    # Preprocess input
    try:
        input_df, encoders = preprocess_input(input_data, feature_columns, categorical_columns)
        
        # Make predictions
        recycling_pred, recycling_prob = make_predictions(rf_recycling, input_df, "Recycling Programs")
        eco_pred, eco_prob = make_predictions(rf_eco, input_df, "Eco-Friendly Manufacturing")
        
        # Display results
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Recycling Programs**")
            st.write(f"Prediction: {'Yes' if recycling_pred == 1 else 'No'}")
            st.write(f"Probability (Yes): {recycling_prob[1]:.2%}")
            st.write(f"Probability (No): {recycling_prob[0]:.2%}")
        
        with col2:
            st.write("**Eco-Friendly Manufacturing**")
            st.write(f"Prediction: {'Yes' if eco_pred == 1 else 'No'}")
            st.write(f"Probability (Yes): {eco_prob[1]:.2%}")
            st.write(f"Probability (No): {eco_prob[0]:.2%}")
        
    
        
    except Exception as e:
        st.error(f"Error processing input: {str(e)}. Please ensure all inputs are valid.")

