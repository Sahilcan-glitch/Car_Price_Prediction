import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# URL to the raw CSV file on GitHub
file_url = 'https://raw.githubusercontent.com/Sahilcan-glitch/car_price_prediction/main/Car%20Price.csv'

# Load the dataset from the URL
car_data = pd.read_csv(file_url)

# Encode categorical variables
car_data = pd.get_dummies(car_data, columns=['Brand', 'Model', 'Fuel', 'Seller_Type', 'Transmission', 'Owner'], drop_first=True)

# Define features and target
X = car_data.drop(['Selling_Price'], axis=1)
y = car_data['Selling_Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Function to predict price of a car in a selected year
def predict_price(car_features, model):
    # Convert car features to DataFrame
    car_features_df = pd.DataFrame([car_features])
    # Encode categorical variables
    car_features_df = pd.get_dummies(car_features_df, drop_first=True)
    # Ensure all columns match the training data
    car_features_df = car_features_df.reindex(columns=X.columns, fill_value=0)
    # Predict price
    predicted_price = model.predict(car_features_df)
    return predicted_price[0]

# Streamlit app
st.title("🚗 Car Price Prediction")

# Input fields
year = st.number_input("Year", min_value=2000, max_value=2023, value=2020, step=1)
brand_list = car_data.columns[car_data.columns.str.startswith('Brand_')].str.replace('Brand_', '').tolist()
brand = st.selectbox("Brand", brand_list)

# Filter models based on selected brand
model_list = car_data.columns[car_data.columns.str.startswith(f'Model_') & car_data.columns.str.contains(brand)].str.replace('Model_', '').tolist()
model_choice = st.selectbox("Model", model_list)

fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel'])

# Prepare input data
car_features = {
    'Year': year,
    f'Brand_{brand}': 1,
    f'Model_{model_choice}': 1,
    f'Fuel_{fuel}': 1
}

# Predict button
if st.button("Predict Price"):
    predicted_price = predict_price(car_features, model)
    st.write(f"The predicted price of the car is: {predicted_price}")

    # Filter data for the selected model
    car_model_data = car_data[(car_data[f'Brand_{brand}'] == 1) & (car_data[f'Model_{model_choice}'] == 1)]
    car_model_data['Year'] = pd.to_numeric(car_model_data['Year'])
    last_10_years_data = car_model_data[car_model_data['Year'] >= year - 10]

    # Create line graph
    fig = px.line(last_10_years_data, x='Year', y='Selling_Price', title=f'Price Change for {model_choice} over the Last 10 Years')
    st.plotly_chart(fig)

# Run the Streamlit app using: streamlit run <script_name>.py
