# ☕ Starbucks Coffee Price Prediction Model

An end-to-end Machine Learning project that predicts the price of Starbucks coffee drinks based on their nutritional characteristics (Calories, Total Fat, Sugars, and Caffeine).

## 📊 Project Overview
This repository contains two trained Machine Learning regression models developed on Kaggle using a Starbucks menu dataset:
1. **Random Forest Regressor** (Best Performing Model)
2. **Decision Tree Regressor**

The models analyze ingredients and beverage categories to predict a competitive price point, mimicking the real-world pricing behavior of coffee chains.

## 📈 Model Performance
* **Random Forest Regressor:** 
  * **Accuracy (R² Score):** 83.41%
  * **Mean Absolute Error (MAE):** $0.18
* **Decision Tree Regressor:** Trained and structured to visualize the rule-based price routing logic.

## 📂 Repository Contents
* `starbucks_random_forest_model.pkl`: Best-fit trained model ready for deployment.
* `starbucks_decision_tree_model.pkl`: Alternative trained decision tree model.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib

## 🚀 Future Scope
The next step for this project is to build an interactive web interface using **Streamlit** or **Flask** so users can input custom coffee ingredients and get real-time price predictions directly from the browser.
