import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def train_car_model():
    df = pd.read_csv("car data.csv")
    df.columns = df.columns.str.strip()

    if "Selling_Price" not in df.columns:
        raise ValueError("Car dataset must contain 'Selling_Price' column")

    # Separate target first
    y = df["Selling_Price"]
    X = df.drop("Selling_Price", axis=1)

    # Encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    with open("car_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("car_columns.pkl", "wb") as f:
        pickle.dump(list(X.columns), f)

    print("Car model trained and saved successfully!")


def train_bike_model():
    df = pd.read_csv("bike data.csv")
    df.columns = df.columns.str.strip()

    target_col = None
    for col in df.columns:
        if col.lower() == "selling_price":
            target_col = col
            break

    if target_col is None:
        raise ValueError("Bike dataset must contain 'selling_price' column")

    y = df[target_col]
    X = df.drop(target_col, axis=1)

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    with open("bike_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("bike_columns.pkl", "wb") as f:
        pickle.dump(list(X.columns), f)

    print("Bike model trained and saved successfully!")


if __name__ == "__main__":
    train_car_model()
    train_bike_model()
    print("Models are ready ✅")