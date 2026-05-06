import pandas as pd

def load_and_clean_data(file_path):

    # Read Excel file
    df = pd.read_excel(file_path)

    # Convert Date column
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort values
    df = df.sort_values(by=['State', 'Date'])

    # Fill missing values
    df['Total'] = df['Total'].fillna(0)

    print("Data Loaded Successfully!")

    print(df.head())

    return df


if __name__ == "__main__":

    from feature_engineering import create_features

    data = load_and_clean_data("../data/Forecasting Case- Study.xlsx")

    data = create_features(data)