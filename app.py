import streamlit as st
import numpy as np
import pandas as pd
import pickle
from graph import plot_year_price_graph

st.set_page_config(
    page_title="Vehicle Resale Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# ---------- CACHE LOADERS ----------
@st.cache_data
def load_data():
    car_df = pd.read_csv("car data.csv")
    bike_df = pd.read_csv("bike data.csv")

    car_df.columns = car_df.columns.str.strip()
    bike_df.columns = bike_df.columns.str.strip()

    return car_df, bike_df


@st.cache_resource
def load_models():
    with open("car_model.pkl", "rb") as f:
        car_model = pickle.load(f)

    with open("car_columns.pkl", "rb") as f:
        car_columns = pickle.load(f)

    with open("bike_model.pkl", "rb") as f:
        bike_model = pickle.load(f)

    with open("bike_columns.pkl", "rb") as f:
        bike_columns = pickle.load(f)

    return car_model, car_columns, bike_model, bike_columns


car_df, bike_df = load_data()
car_model, car_columns, bike_model, bike_columns = load_models()

# ---------- HELPERS ----------
def create_input_df(columns):
    return pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

def set_value(input_df, col_name, value):
    if col_name in input_df.columns:
        input_df.at[0, col_name] = value

def get_suggestion(predicted_price, current_price):
    if predicted_price < current_price * 0.7:
        return "This looks overpriced compared to the predicted resale value."
    elif predicted_price > current_price * 0.9:
        return "This looks like a strong and fair resale value."
    else:
        return "This looks like a reasonable resale price."

def find_column_case_insensitive(df, target_name):
    for col in df.columns:
        if col.strip().lower() == target_name.strip().lower():
            return col
    return None


# ---------- MAIN PAGE ----------
st.title("🚗🏍️ Vehicle Resale Price Prediction System")
st.write("Check the estimated resale value of your car or bike in a simple and quick way.")

role = st.selectbox("Select User Role", ["Customer", "Seller"])
vehicle_type = st.selectbox("Select Vehicle Type", ["Car", "Bike"])

# ===================== CAR SECTION =====================
if vehicle_type == "Car":
    st.subheader("🚗 Car Details")

    car_names = sorted(car_df["Car_Name"].dropna().unique().tolist())
    car_name_option = st.selectbox("Select Car Name", car_names + ["Other"])

    custom_car_name = ""
    if car_name_option == "Other":
        custom_car_name = st.text_input("Type Car Name")

    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input("Year", min_value=2000, max_value=2025, value=2018)
        present_price = st.number_input("Current / Present Price (in lakhs)", min_value=0.0, value=5.0)
        kms_driven = st.number_input("KMs Driven", min_value=0, value=30000)

    with col2:
        owner = st.selectbox("Owner", [0, 1, 2, 3], index=0)
        fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
        seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

    if st.button("Predict Car Price"):
        input_df = create_input_df(car_columns)

        set_value(input_df, "Year", year)
        set_value(input_df, "Present_Price", present_price)
        set_value(input_df, "Kms_Driven", kms_driven)
        set_value(input_df, "Owner", owner)

        fuel_col = f"Fuel_Type_{fuel}"
        set_value(input_df, fuel_col, 1)

        seller_col = f"Seller_Type_{seller_type}"
        set_value(input_df, seller_col, 1)

        transmission_col = f"Transmission_{transmission}"
        set_value(input_df, transmission_col, 1)

        final_car_name = car_name_option if car_name_option != "Other" else custom_car_name.strip()

        if final_car_name:
            car_name_col = f"Car_Name_{final_car_name}"
            if car_name_col in input_df.columns:
                set_value(input_df, car_name_col, 1)
            elif car_name_option == "Other":
                st.warning("This car name was not present in training data, so brand effect is not included in prediction.")

        predicted_price = car_model.predict(input_df)[0]
        predicted_price = max(0, round(float(predicted_price), 2))

        suggestion = get_suggestion(predicted_price, present_price)

        st.success(f"Predicted Car Resale Price: ₹ {predicted_price} Lakhs")

        if role == "Customer":
            st.info(f"Customer View: {suggestion}")
        else:
            st.info(f"Seller View: Estimated expected resale value generated successfully. {suggestion}")

        st.subheader("📊 Car Price Trend Graph")
        fig = plot_year_price_graph(car_df, "Year", "Selling_Price", "Car Year vs Selling Price")
        st.pyplot(fig)

# ===================== BIKE SECTION =====================
elif vehicle_type == "Bike":
    st.subheader("🏍️ Bike Details")

    bike_name_col = find_column_case_insensitive(bike_df, "name")
    year_col = find_column_case_insensitive(bike_df, "year")
    km_col = find_column_case_insensitive(bike_df, "km_driven")
    seller_col = find_column_case_insensitive(bike_df, "seller_type")
    owner_col = find_column_case_insensitive(bike_df, "owner")
    ex_price_col = find_column_case_insensitive(bike_df, "ex_showroom_price")
    selling_price_col = find_column_case_insensitive(bike_df, "selling_price")

    bike_names = sorted(bike_df[bike_name_col].dropna().unique().tolist()) if bike_name_col else []
    bike_name_option = st.selectbox("Select Bike Name", bike_names + ["Other"])

    custom_bike_name = ""
    if bike_name_option == "Other":
        custom_bike_name = st.text_input("Type Bike Name")

    col1, col2 = st.columns(2)

    with col1:
        bike_year = st.number_input("Bike Year", min_value=2000, max_value=2025, value=2020)
        bike_price = st.number_input("Current / Ex-showroom Price", min_value=0.0, value=80000.0)
        bike_km = st.number_input("Bike KMs Driven", min_value=0, value=10000)

    with col2:
        if owner_col:
            owner_values = sorted(bike_df[owner_col].dropna().astype(str).unique().tolist())
            bike_owner = st.selectbox("Owner", owner_values)
        else:
            bike_owner = ""

        if seller_col:
            seller_values = sorted(bike_df[seller_col].dropna().astype(str).unique().tolist())
            bike_seller = st.selectbox("Seller Type", seller_values)
        else:
            bike_seller = ""

    if st.button("Predict Bike Price"):
        input_df = create_input_df(bike_columns)

        if year_col:
            set_value(input_df, year_col, bike_year)

        if km_col:
            set_value(input_df, km_col, bike_km)

        if ex_price_col:
            set_value(input_df, ex_price_col, bike_price)

        if owner_col and bike_owner:
            owner_dummy = f"{owner_col}_{bike_owner}"
            set_value(input_df, owner_dummy, 1)

        if seller_col and bike_seller:
            seller_dummy = f"{seller_col}_{bike_seller}"
            set_value(input_df, seller_dummy, 1)

        final_bike_name = bike_name_option if bike_name_option != "Other" else custom_bike_name.strip()

        if bike_name_col and final_bike_name:
            bike_name_dummy = f"{bike_name_col}_{final_bike_name}"
            if bike_name_dummy in input_df.columns:
                set_value(input_df, bike_name_dummy, 1)
            elif bike_name_option == "Other":
                st.warning("This bike name was not present in training data, so brand effect is not included in prediction.")

        predicted_price = bike_model.predict(input_df)[0]
        predicted_price = max(0, round(float(predicted_price), 2))

        suggestion = get_suggestion(predicted_price, bike_price)

        st.success(f"Predicted Bike Resale Price: ₹ {predicted_price}")

        if role == "Customer":
            st.info(f"Customer View: {suggestion}")
        else:
            st.info(f"Seller View: Estimated expected resale value generated successfully. {suggestion}")

        if year_col and selling_price_col:
            st.subheader("📊 Bike Price Trend Graph")
            fig = plot_year_price_graph(bike_df, year_col, selling_price_col, "Bike Year vs Selling Price")
            st.pyplot(fig)
        else:
            st.warning("Bike graph cannot be shown because required columns were not found in dataset.")

# ---------- USEFULNESS SECTION AT BOTTOM ----------
st.markdown("---")
st.subheader("About this Application!")

st.write("""
This application helps users make smarter decisions before buying or selling a vehicle.

For a customer, it gives an estimated resale value so they can check whether the quoted price looks fair or not.
For a seller, it helps in understanding an expected market value before listing the vehicle.

It reduces dependency on guesswork and gives a quick price estimate based on vehicle type, brand, year, distance driven, ownership, and other important factors.

The graph gives extra clarity by showing how vehicle prices generally change over time, helping users understand the trend instead of only seeing one predicted value.
""")
