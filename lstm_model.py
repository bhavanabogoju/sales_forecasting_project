import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Load dataset
df = pd.read_excel("../data/Forecasting Case- Study.xlsx")


# Use Total column
data = df['Total'].values.reshape(-1, 1)


# Scale data
scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)


# Create sequences
X = []
y = []

sequence_length = 5

for i in range(sequence_length, len(scaled_data)):

    X.append(scaled_data[i-sequence_length:i])

    y.append(scaled_data[i])


X = np.array(X)

y = np.array(y)


# Train-test split
train_size = int(len(X) * 0.8)

X_train = X[:train_size]

X_test = X[train_size:]

y_train = y[:train_size]

y_test = y[train_size:]


# Build LSTM model
model = Sequential()

model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))

model.add(Dense(1))


# Compile model
model.compile(optimizer='adam', loss='mse')


# Train model
model.fit(X_train, y_train, epochs=10, batch_size=16)


# Predictions
predictions = model.predict(X_test)


# Convert back to original values
predictions = scaler.inverse_transform(predictions)

y_test_actual = scaler.inverse_transform(y_test)


# RMSE
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))


print("LSTM Model Completed!")

print("LSTM RMSE:", rmse)