# stock_prediction.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import yfinance as yf

# Download stock data (e.g., Apple)
data = yf.download('AAPL', start='2015-01-01', end='2024-01-01')
data['Date'] = data.index
data['Days'] = (data['Date'] - data['Date'].min()).days

# Feature and target
X = np.array(data['Days']).reshape(-1, 1)
y = data['Close']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Root Mean Squared Error: {rmse:.2f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(data['Days'], y, label='Actual Price', color='gray', alpha=0.4)
plt.plot(X_test, y_pred, color='red', label='Predicted Line')
plt.xlabel('Days since start')
plt.ylabel('Stock Price')
plt.title('Stock Market Prediction (Linear Regression)')
plt.legend()
plt.show()
