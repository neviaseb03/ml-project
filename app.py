import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Initialize the Flask application
app = Flask(__name__)

# Read the dataset
data = pd.read_csv('flight_delay_predict.csv')

# Remove rows with missing values
data.dropna(axis=0, inplace=True)

# Convert 'FlightDate' to datetime
data['FlightDate'] = pd.to_datetime(data['FlightDate'])

# Split the data into features (X) and target (y)
X = data[['AirTime', 'Distance']]
y = data['ArrDelayMinutes']  # Use 'ArrDelayMinutes' as the target for regression

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    input_data = request.json
    air_time = input_data['AirTime']
    distance = input_data['Distance']
    
    # Create an array from the input data
    user_input = np.array([[air_time, distance]])
    
    # Scale the input data
    user_input_scaled = scaler.transform(user_input)
    
    # Make the prediction
    prediction = model.predict(user_input_scaled)[0]
    
    # Return the prediction as a JSON response
    return jsonify({'Predicted Arrival Delay (minutes)': prediction})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
