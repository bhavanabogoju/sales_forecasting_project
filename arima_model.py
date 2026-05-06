import pandas as pd

from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error

import numpy as np


# Load dataset
df = pd.read_excel("../data/Forecasting Case- Study.xlsx")


# Convert date
df['Date'] = pd.to_datetime(df['Date'])


# Sort data
df = df.sort_values(by='Date')


# Use Total column
sales = df['Total']


# Train-test split
train_size = int(len(sales) * 0.8)

train = sales[:train_size]

test = sales[train_size:]


# Create ARIMA model
model = ARIMA(train, order=(1,1,1))


# Train model
model_fit = model.fit()


# Forecast
predictions = model_fit.forecast(steps=len(test))


# RMSE
rmse = np.sqrt(mean_squared_error(test, predictions))


print("ARIMA Model Completed!")

print("ARIMA RMSE:", rmse)