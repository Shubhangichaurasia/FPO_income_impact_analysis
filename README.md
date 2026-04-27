# 🌾 SAT-AGRI: FPO Impact & Benefit Analyzer

## 🚀 Live Demo

🔗 https://fpo-project-ishwlu45jxhg5chafegtnn.streamlit.app/

This is a fully deployed interactive dashboard built using Streamlit, allowing real-time prediction and analysis of FPO impact.

## 📌 Project Overview

SAT-AGRI is an end-to-end machine learning and analytics system designed to evaluate and predict the impact of Farmer Producer Organizations (FPOs) on agricultural productivity and farmer income.

The system combines predictive modeling, causal inference, and interactive visualization to provide actionable insights for farmers.

---

## 🚀 Key Features

### 1. Predictive Analytics

* Predicts farmer income, yield, and cultivation cost
* Compares scenarios:

  * Without FPO
  * With FPO

### 2. Causal Impact Analysis

* Uses Linear Regression to estimate direct financial impact
* Isolates effect of FPO participation

### 3. Recommendation Engine

* Suggests improvements like:

  * Soil health improvement
  * Access to credit
  * Mechanization

### 4. Interactive Dashboard

* Built using Streamlit
* Real-time input and predictions
* Visual insights using Matplotlib and Seaborn

---

## 🧠 Machine Learning Model

* Model Used: Gradient Boosting Regressor
* Target Variable: Income_Annual
* Features:

  * Land size, rainfall, soil health
  * Access to credit, tractor ownership
  * Crop type, district
* Evaluation Metric:

  * RMSE ≈ ₹4700

---

## 📊 Dataset

* Synthetic dataset simulating real-world farming scenarios
* Includes:

  * Treatment group (FPO members)
  * Control group (non-members)
* Features:

  * Environmental (rainfall, temperature)
  * Socio-economic (credit, income)
  * Agricultural (yield, cost)

---

## 🛠️ Data Processing

### Data Cleaning

* Missing values handled using KNN Imputation
* Outliers treated using IQR method

### Feature Engineering

* One-hot encoding for categorical variables
* NDVI-based crop health metrics

---

## 🏗️ Project Structure

```
├── app.py                  # Streamlit dashboard
├── train_model.py          # Model training script
├── dataset.py              # Synthetic dataset generation
├── clean_data.py           # Data cleaning pipeline
├── check_data.py           # Data validation
├── fpo_data_with_ndvi.csv  # Final dataset
├── fpo_predictor_model.joblib
├── model_columns.joblib
```

---

## ⚙️ How to Run

1. Train the model:

```
python train_model.py
```

2. Run the dashboard:

```
streamlit run app.py
```

---

## 📈 Key Insights

* FPO participation leads to:

  * Increased income
  * Higher crop yield
  * Reduced cultivation cost
* Satellite NDVI confirms improvement in farm health

---

## 🎯 Conclusion

This project demonstrates how data science and machine learning can be used to:

* Improve agricultural decision-making
* Quantify policy impact
* Provide personalized recommendations to farmers

---

## 🔮 Future Improvements

* Use real-world datasets
* Add deep learning models
* Integrate live satellite data APIs
* Deploy on cloud (AWS/GCP)

---
