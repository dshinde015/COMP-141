print("✅ Script is running!")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 1: Download data
ticker = "AAPL"
df = yf.download(ticker, period="6mo")
df = df[["Close"]]

# Step 2: Create target
df["Target"] = df["Close"].shift(-1)
df.dropna(inplace=True)

# Step 3: Prepare data
X = df[["Close"]]
y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict and print
today_price = df["Close"].iloc[-1]
predicted_price = model.predict(np.array([today_price]).reshape(-1, 1))
print(f"Predicted next price for {ticker}: ${predicted_price[0]:.2f}")

# Step 6: Plot results
plt.plot(y_test.values, label="Actual")
plt.plot(model.predict(X_test), label="Predicted")
plt.legend()
plt.title(f"{ticker} Stock Price Prediction")
plt.show()

