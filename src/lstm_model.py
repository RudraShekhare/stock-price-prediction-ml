import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# Load the dataset
df = pd.read_csv("data/features.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Use only the Close column
data = df[["Close"]].values

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Prepare training data (60 timesteps)
X = []
y = []

for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# Reshape for LSTM (samples, timesteps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build the model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32)

# Predict on training data (for demo)
predictions = model.predict(X)
predicted_prices = scaler.inverse_transform(predictions)
real_prices = scaler.inverse_transform(y.reshape(-1, 1))

# Plot the predictions
plt.figure(figsize=(10, 6))
plt.plot(real_prices, label="Real Prices")
plt.plot(predicted_prices, label="Predicted Prices")
plt.title("LSTM Model Prediction (Training Data)")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()

# Save plot
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/lstm_predictions.png")
print("✅ Plot saved to plots/lstm_predictions.png")

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/lstm_model.h5")
print("✅ Model saved to models/lstm_model.h5")
