from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load(r'model/salary_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        edu = float(request.form['educational_num'])
        hours = float(request.form['hours_per_week'])

        features = np.array([[age, edu, hours]])

        prediction = model.predict(features)

        return render_template(
            'index.html',
            prediction_text=f'Predicted Salary Class: {prediction[0]}'
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f'Error: {str(e)}'
        )

if __name__ == '__main__':
    app.run(debug=True)