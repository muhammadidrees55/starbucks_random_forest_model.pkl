
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration & Theme Styling
st.set_page_config(
    page_title="Starbucks Price Predictor",
    page_icon="☕",
    layout="centered"
)

# Custom Styling for Starbucks feel
st.markdown("""
    <style>
    .main { background-color: #f7f7f7; }
    h1 { color: #00704A; }
    .stButton>button { background-color: #00704A; color: white; border-radius: 8px; }
    .stButton>button:hover { background-color: #004d33; color: white; }
    </style>
""", unsafe_allow_html=True)

# 2. Load the Best Performing Trained Model
@st.cache_resource
def load_model():
    return joblib.load('starbucks_random_forest_model.pkl')

try:
    model = load_model()
    # Extract the exact feature names the model was trained on
    model_features = list(model.feature_names_in_)
except Exception as e:
    st.error("⚠️ Model file not found or failed to load. Make sure 'starbucks_random_forest_model.pkl' is in the repository root.")
    st.stop()

# 3. Web App Header
st.title("☕ Starbucks Coffee Price Predictor")
st.write("Predict the price of a custom coffee beverage based on its characteristics using our trained **Random Forest Regressor**.")
st.markdown("---")

# 4. User Inputs via Sidebar/Form
st.header("🔧 Customize Your Beverage")

category = st.selectbox(
    "Choose Beverage Category:",
    [
        "Classic Espresso Drinks", "Coffee", "Frappuccino® Blended Coffee", 
        "Frappuccino® Blended Crème", "Frappuccino® Light Blended Coffee", 
        "Shaken Iced Beverages", "Smoothies", "Tazo® Tea Drinks"
    ]
)

calories = st.slider("Calories (kcal):", min_value=0, max_value=600, value=150, step=5)
total_fat = st.slider("Total Fat (g):", min_value=0.0, max_value=20.0, value=3.0, step=0.5)
sugars = st.slider("Sugars (g):", min_value=0, max_value=100, value=25, step=1)
caffeine = st.slider("Caffeine (mg):", min_value=0, max_value=500, value=75, step=5)

# 5. Prediction Logic
if st.button("🔮 Predict Coffee Price"):
    # Create a base dictionary with user inputs
    input_data = {
        'Calories': float(calories),
        'Total Fat (g)': float(total_fat),
        'Sugars (g)': float(sugars),
        'Caffeine (mg)': float(caffeine)
    }
    
    # Process One-Hot Encoded categories dynamic match
    target_category_col = f"Beverage_category_{category}"
    
    # Dynamically build the DataFrame using the exact column names from the model
    final_input_dict = {}
    for col in model_features:
        if col in input_data:
            final_input_dict[col] = [input_data[col]]
        elif col == target_category_col:
            final_input_dict[col] = [1]
        else:
            final_input_dict[col] = [0]
            
    # Convert to DataFrame with the perfect column sequence
    input_df = pd.DataFrame(final_input_dict)
    input_df = input_df[model_features]
    
    # Perform prediction safely without index mismatch
    predicted_price = model.predict(input_df)[0]
    
    # Display Result
    st.markdown("---")
    st.success(f"### 💰 Estimated Starbucks Price: **${predicted_price:.2f}**")
    st.info("💡 *Pricing is estimated based on the drink's nutritional profile and category characteristics.*")
