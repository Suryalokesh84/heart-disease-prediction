from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the trained model
model = LogisticRegression()
heart_data = pd.read_csv('heart_1.csv')
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
model.fit(X, Y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extracting features from form submission
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach']
    input_data = [float(request.form[feature]) for feature in features]
    
    # Make prediction
    prediction = model.predict([input_data])[0]
    
    # Returning prediction as HTML response
    if prediction == 0:
        result = 'The Person does not have a Heart Disease'
    else:
        result = 'The Person has Heart Disease'
    
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
