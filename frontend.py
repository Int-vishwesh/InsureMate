import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- 1. SETUP PAGE ---
st.set_page_config(page_title="InsureMate", layout="centered")

st.title("Health Insurance Premium Prediction")
st.markdown("Enter the details below to predict your health insurance premium category:")

# --- 2. LOAD THE MODEL (The "Brain") ---
# We use @st.cache_resource so it only loads once, making the app faster
@st.cache_resource
def load_model():
    try:
        return pickle.load(open('model.pkl', 'rb'))
    except FileNotFoundError:
        st.error("Error: 'model.pkl' not found. Make sure it's in the same folder as this file.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- 3. UI INPUTS ---
age = st.number_input("Age", min_value=1, max_value=120, value=30)
weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0)
height = st.number_input("Height (m)", min_value=0.5, max_value=3.0, value=1.7)
income_lpa = st.number_input("Annual Income (LPA)", min_value=0.0, max_value=1000.0, value=5.0)
smoker = st.selectbox("Are you a Smoker?", options=[True, False])

# Categorical options
city_options = [
    "Delhi", "Mumbai", "Bengaluru", "Chennai", "Hyderabad", "Kolkata", "Pune", "Ahmedabad",
    "Chandigarh", "Jaipur", "Lucknow", "Indore", "Nagpur", "Kochi", "Coimbatore",
    "Bhubaneswar", "Surat", "Vadodara", "Bhopal", "Ludhiana", "Kanpur", "Patna",
    "Agra", "Amritsar", "Varanasi", "Guwahati", "Raipur", "Ranchi", "Visakhapatnam",
    "Mangalore", "Patiala", "Dehradun", "Udaipur", "Jodhpur", "Guntur", "Mysore",
    "Rajkot", "Madurai", "Allahabad", "Aurangabad", "Jalandhar", "Kolhapur",
    "Trivandrum", "Gwalior", "Jamshedpur", "Bareilly", "Dhanbad", "Siliguri"
]

occupation_options = [
    'retired', 'freelancer', 'student', 'government_job',
    'business_owner', 'unemployed', 'private_job'
]

city = st.selectbox("City", options=city_options)
occupation = st.selectbox("Occupation", options=occupation_options)

# --- 4. PREDICTION LOGIC ---
if st.button("Predict Premium Category"):
    if model is None:
        st.error("Model is not loaded. Please check the file.")
    else:
        # Create a DataFrame (Same format as the model was trained on)
        input_data = pd.DataFrame([{
            "age": age,
            "weight": weight,
            "height": height,
            "income_lpa": income_lpa,
            "smoker": 1 if smoker else 0, # Convert Boolean to 1/0 just in case
            "city": city,
            "occupation": occupation
        }])

        try:
            # Direct Prediction
            prediction = model.predict(input_data)
            
            # Show Result
            # We assume the prediction returns an array, so we take the first item
            final_result = prediction[0]
            st.success(f"Predicted Premium Category: {final_result}")
            
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
            st.markdown("### Troubleshooting:")
            st.info("If you see an error about 'could not convert string to float', it means your model needs numbers (Label Encoding) but we sent text (Delhi/Student).")
            st.info("If that happens, we need to copy the 'mapping' logic from your main.py file.")
