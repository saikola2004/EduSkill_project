from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Initialize model
model = LinearRegression()

def train_model():
    # Load and preprocess your data
    data = pd.read_csv('TSLA.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    # Create features and labels
    data['Previous_Close'] = data['Close'].shift(1)
    data.dropna(inplace=True)

    # Define features and target
    X = data[['Previous_Close']]
    y = data['Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Parse incoming JSON data
    previous_close = float(data['previous_close'])
    
    # Make prediction
    prediction = model.predict([[previous_close]])[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    train_model()  # Train the model when the app starts
    app.run(debug=True)
