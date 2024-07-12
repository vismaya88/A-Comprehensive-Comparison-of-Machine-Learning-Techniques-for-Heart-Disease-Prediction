from flask import Flask, render_template, request
import pandas as pd
#from sklearn.preprocessing import StandardScaler
from joblib import load

app = Flask(__name__)

# Load the trained model
model = load("heart_disease_classifier.joblib")
scaler = load("scaler.joblib")

# Define feature names
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

@app.route('/')
def index():
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from the form
    input_features = [float(request.form[feature]) for feature in feature_names]
    # Convert input features to DataFrame
    input_df = pd.DataFrame([input_features], columns=feature_names)
    # Perform feature scaling
    #scaler = StandardScaler()
    input_scaled = scaler.transform(input_df)
    # Predict using the model
    prediction = model.predict(input_scaled)
    result = "Heart Disease" if prediction[0] == 1 else "No Heart Disease"
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
