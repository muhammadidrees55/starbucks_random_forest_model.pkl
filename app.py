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
    # Loading the Random Forest Model which had 83.41% accuracy
    return joblib.load('starbucks_random_forest_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error("⚠️ Model file not found. Make sure 'starbucks_random_forest_model.pkl' is in the repository root.")
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
    # Create matching features dictionary for the model
    input_data = {
        'Calories': [float(calories)],
        'Total Fat (g)': [float(total_fat)],
        'Sugars (g)': [float(sugars)],
        'Caffeine (mg)': [float(caffeine)]
    }
    
    input_df = pd.DataFrame(input_data)
    
    # Target columns expected by One-Hot Encoding during training
    expected_categories = [
        'Beverage_category_Coffee',
        'Beverage_category_Frappuccino® Blended Coffee',
        'Beverage_category_Frappuccino® Blended Crème',
        'Beverage_category_Frappuccino® Light Blended Coffee',
        'Beverage_category_Shaken Iced Beverages',
        'Beverage_category_Smoothies',
        'Beverage_category_Tazo® Tea Drinks'
    ]
    
    # Initialize all category columns to 0
    for col in expected_categories:
        input_df[col] = 0
        
    # Set the selected category column to 1 (excluding the base category 'Classic Espresso Drinks')
    target_col = f"Beverage_category_{category}"
    if target_col in expected_categories:
        input_df[target_col] = 1
        
    # Reorder columns to match exact model training shape
    # This matches the structure of 4 base features + 7 categorical dummies
    feature_order = ['Calories', 'Total Fat (g)', 'Sugars (g)', 'Caffeine (mg)'] + expected_categories
    input_df = input_df[feature_order]
    
    # Perform prediction
    predicted_price = model.predict(input_df)[0]
    
    # Display Result
    st.markdown("---")
    st.success(f"### 💰 Estimated Starbucks Price: **${predicted_price:.2f}**")
    st.info("💡 *Pricing is estimated based on the drink's nutritional profile and category characteristics.*")
