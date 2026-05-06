import pandas as pd

from prophet import Prophet

from sklearn.metrics import mean_squared_error

import numpy as np


# Load dataset
df = pd.read_excel("../data/Forecasting Case- Study.xlsx")


# Convert columns for Prophet
df = df.rename(columns={
    'Date': 'ds',
    'Total': 'y'
})


# Keep required columns only
df = df[['ds', 'y']]


# Convert date
df['ds'] = pd.to_datetime(df['ds'])


# Train-test split
train_size = int(len(df) * 0.8)

train = df[:train_size]

test = df[train_size:]


# Create Prophet model
model = Prophet()


# Train model
model.fit(train)


# Future dates
future = model.make_future_dataframe(periods=len(test))


# Forecast
forecast = model.predict(future)


# Predictions
predictions = forecast['yhat'][-len(test):].values


# RMSE
rmse = np.sqrt(mean_squared_error(test['y'], predictions))


print("Prophet Model Completed!")

print("Prophet RMSE:", rmse)