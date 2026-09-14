# 📊 Colliers AI | Mass Appraisal & XAI Dashboard

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1793D1?style=for-the-badge&logo=xgboost&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## Project Overview
This repository contains a **Proof of Concept (PoC)** dashboard developed for the Data Analyst / Engineer position at **Colliers Italia**. The application is a Mass Real Estate Appraisal tool that leverages tree-based Machine Learning (**XGBoost**) and Explainable AI (**SHAP**) to provide transparent, accurate, and scalable property valuations.

The primary goal of this dashboard is to move beyond "black-box" valuation models. By integrating Global and Local Explainability mechanisms, it empowers both technical and non-technical stakeholders to understand exactly *why* a specific property is valued at a certain price.

## 🚀 Key Features

*   **Macro Performance & Feature Importance:** Showcases the champion model's metrics ($R^2$, RMSE, MAPE) outperforming traditional OLS methods. A SHAP Bar Plot dynamically highlights which features positively (Green) or negatively (Red) impact the overall market.
*   **Micro XAI & "What-If" Simulator:** An interactive simulation engine where users can tweak 11 property parameters (e.g., Gross Area, Building Age, Distance to Bazaars) to see real-time price changes.
*   **Natural Language Explanation (NLE):** Translates complex SHAP values into business-friendly, readable text summaries.
*   **Waterfall Price Breakdown:** Visually decomposes the final estimated price step-by-step from the market base value.

## 📁 Dataset & Methodology
*   **Region:** Mersin / Yenişehir, Turkey.
*   **Data Points:** 1,181 residential properties.
*   **Features Used:** 11 critical value determinants (Gross Area, Net Area, Distance to Bazaars, Distance to Coach Station, Building Floors, Floor Number, Number of Bathrooms, Number of Rooms, Building Age, Elevation, Heating System).
*   **Model:** `XGBRegressor` optimized for predictive accuracy and combined with `shap.TreeExplainer` for interpretability.

---

## 🛠️ How to run it on your own machine

### Option 1: Standard Python (Pip)
1. Clone the repository and navigate to the project folder.
2. Install the required dependencies:
   ```bash
   pip install streamlit pandas numpy xgboost shap matplotlib openpyxl
