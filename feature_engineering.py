import pandas as pd

def create_features(df):

    # Month feature
    df['month'] = df['Date'].dt.month

    # Week feature
    df['week'] = df['Date'].dt.isocalendar().week

    # Lag features
    df['lag_1'] = df.groupby('State')['Total'].shift(1)

    df['lag_7'] = df.groupby('State')['Total'].shift(7)

    # Rolling mean
    df['rolling_mean'] = (
        df.groupby('State')['Total']
        .transform(lambda x: x.rolling(4).mean())
    )

    # Remove empty rows
    df = df.dropna()

    print("Features Created Successfully!")

    print(df.head())

    return df