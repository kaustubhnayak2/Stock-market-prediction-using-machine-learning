# 📈 Stock Market Price Prediction with Linear Regression

This project uses **Linear Regression** to predict stock prices using historical stock market data fetched through the **Yahoo Finance API (yfinance)**.

---

## 🚀 Project Overview

- **Stock**: Apple Inc. (AAPL)
- **Data Source**: [Yahoo Finance](https://finance.yahoo.com/)
- **Algorithm**: Linear Regression (via Scikit-Learn)
- **Goal**: Predict closing stock prices and visualize the regression line

---

## 📊 Libraries Used

- `pandas` – Data handling
- `numpy` – Numerical computation
- `matplotlib` – Visualization
- `scikit-learn` – Machine learning (Linear Regression)
- `yfinance` – Fetching historical stock data

---

## 🧠 Workflow

1. **Download stock data** for Apple (AAPL) from 2015 to 2024 using `yfinance`
2. **Convert date to numerical features** for regression modeling
3. **Split the dataset** into training and testing sets
4. **Train a Linear Regression model**
5. **Predict and evaluate** using RMSE
6. **Visualize** the actual vs. predicted stock prices

---

## 🖥️ How to Run the Code

### 🔧 Prerequisites

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn yfinance
