from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from pdfminer.high_level import extract_text
import re
import io

# Initialize Flask app
app = Flask(__name__)

# Sample Data (same as the training data)
data = {
    'Age': [45, 50, 37, 62, 29],
    'Cholesterol (mg/dL)': [220, 250, 190, 230, 180],
    'Blood Pressure (mm Hg)': [130, 140, 120, 135, 125],
    'Glucose (mg/dL)': [100, 110, 90, 105, 95],
    'BMI': [24.5, 28.7, 22.1, 29.5, 21.0],
    'Disease': [1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

scaler = MinMaxScaler()
X = scaler.fit_transform(df[['Age', 'Cholesterol (mg/dL)', 'Blood Pressure (mm Hg)', 'Glucose (mg/dL)', 'BMI']])
y = df['Disease']

model = LogisticRegression()
model.fit(X, y)

def extract_text_from_pdf(pdf_file):
    try:
        # Read and extract text from the uploaded PDF file
        pdf_data = io.BytesIO(pdf_file.read())
        text = extract_text(pdf_data)
        return text
    except Exception as e:
        return None


def extract_values_from_text(text):
    age = re.search(r'Age:\s*(\d+)', text)
    cholesterol = re.search(r'Cholesterol:\s*(\d+)', text)
    bp = re.search(r'Blood Pressure:\s*(\d+)', text)
    glucose = re.search(r'Glucose:\s*(\d+)', text)
    bmi = re.search(r'BMI:\s*([\d.]+)', text)

    if age and cholesterol and bp and glucose and bmi:
        return float(age.group(1)), float(cholesterol.group(1)), float(bp.group(1)), float(glucose.group(1)), float(bmi.group(1))
    else:
        return None


@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    age = None
    cholesterol = None
    bp = None
    glucose = None
    bmi = None

    
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        
        text = extract_text_from_pdf(file)
        if text:
            
            values = extract_values_from_text(text)
            if values:
                age, cholesterol, bp, glucose, bmi = values
            else:
                result = "Could not extract values from the PDF. Please ensure it follows the expected format."
                return render_template('chat.html', prediction_text=result)
        else:
            result = "Error reading the PDF file."
            return render_template('chat.html', prediction_text=result)

    else:
        
        try:
            age = float(request.form['age'])
            cholesterol = float(request.form['cholesterol'])
            bp = float(request.form['bp'])
            glucose = float(request.form['glucose'])
            bmi = float(request.form['bmi'])
        except ValueError:
            result = "Please enter valid numerical values in the form."
            return render_template('chat.html', prediction_text=result)

    if age is not None and cholesterol is not None and bp is not None and glucose is not None and bmi is not None:
        
        input_data = np.array([[age, cholesterol, bp, glucose, bmi]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        if prediction == 1:
            result = "Based on your test results, you may be at risk for the disease. Please consult with a doctor."
        else:
            result = "Your test results look good. However, it's always a good idea to consult with a doctor."

        return render_template('chat.html', prediction_text=result)
    else:
        result = "Please provide valid data through either a form or a PDF file."
        return render_template('chat.html', prediction_text=result)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
