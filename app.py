from fastapi import FastAPI

import pandas as pd
import joblib


app = FastAPI()


# Load trained model
model = joblib.load("models/xgboost_model.pkl")


@app.get("/predict")

def predict(
    month: int,
    week: int,
    lag_1: float,
    lag_7: float,
    rolling_mean: float
):

    data = pd.DataFrame([{
        "month": month,
        "week": week,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "rolling_mean": rolling_mean
    }])

    prediction = model.predict(data)

    return {
        "forecast": prediction.tolist()
    }
    