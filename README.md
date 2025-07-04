# 📈 Stock Price Prediction using Machine Learning

This project aims to predict future stock prices using historical data and machine learning techniques. The goal is to create a reproducible and understandable model pipeline that can be used to forecast the closing price of a stock (e.g., Apple Inc.).

---

## 📊 Problem Statement

Stock price prediction is a common problem in quantitative finance. This project focuses on using historical stock data and machine learning models to predict the next day's closing price.

---

## 📂 Project Structure
stock-price-prediction-ml/
├── data/ # Raw and processed stock data
├── notebooks/ # Jupyter notebooks for exploration and modeling
├── src/ # Python scripts for core logic
├── models/ # Saved models
├── README.md # Project documentation
└── requirements.txt # Python dependencies

## 📌 Features

- Data collection using `yfinance`
- Feature engineering (moving averages, lag features)
- ML models: Linear Regression, Random Forest (more to be added)
- Model evaluation (MAE, MSE, R²)
- Visualizations of actual vs predicted prices
- Reproducible project structure

---

## 📈 Models Used

- Linear Regression (baseline)
- Random Forest Regressor
- (Optional: LSTM for time series prediction)
