from flask import Flask, render_template, request
import os
import numpy as np
import pickle

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to models
diagnosis_model_path = os.path.join(BASE_DIR, "et_model_diagnosis.pkl")
severity_model_path = os.path.join(BASE_DIR, "rf_model_severity.pkl")

# Load models
diagnosis_model = pickle.load(open(diagnosis_model_path, 'rb'))
severity_model = pickle.load(open(severity_model_path, 'rb'))

# Encodings
diagnosis_map = {
    'Asthma': 0,
    'Chronic Kidney Disease (CKD)': 1,
    'Heart Failure': 2,
    'Myocardial Infarction': 3,
    'Normal': 4,
    'Pneumonia': 5,
    'Sepsis': 6,
    'Stroke': 7
}
severity_map = {'Mild': 0, 'Moderate': 1, 'Severe': 2}

reverse_diagnosis_map = {v: k for k, v in diagnosis_map.items()}
reverse_severity_map = {v: k for k, v in severity_map.items()}


@app.route('/', methods=['GET', 'POST'])
def disease_predict():
    diagnosis_result = None

    if request.method == 'POST':
        data = request.form

        age = float(data['age'])
        height = float(data['height'])
        weight = float(data['weight'])
        systolic_bp = float(data['systolic_bp'])
        diastolic_bp = float(data['diastolic_bp'])
        glucose = float(data['glucose'])
        cholesterol = float(data['cholesterol'])
        creatinine = float(data['creatinine'])
        diabetes = int(data['diabetes'])
        hypertension = int(data['hypertension'])
        sex = data['sex']

        # Derived features
        bmi = weight / ((height / 100) ** 2)
        bmi_category = (
            "Underweight" if bmi < 18.5 else
            "Normal" if bmi < 25 else
            "Overweight" if bmi < 30 else "Obese"
        )

        MAP = (systolic_bp + 2 * diastolic_bp) / 3
        comorbidity_count = diabetes + hypertension
        high_creatinine_flag = 1 if creatinine > 1.2 else 0
        hyperglycemia_flag = 1 if glucose > 140 else 0
        abnormal_bp_flag = 1 if systolic_bp > 140 or diastolic_bp > 90 else 0
        age_creatinine_risk = 1 if (age > 60 and creatinine > 1.2) else 0

        sex_Male = int(sex == 'Male')
        sex_Other = int(sex == 'Other')
        BMI_Category_Obese = int(bmi_category == 'Obese')
        BMI_Category_Overweight = int(bmi_category == 'Overweight')
        BMI_Category_Underweight = int(bmi_category == 'Underweight')

        # Model input
        input_features = np.array([[
            age, bmi, systolic_bp, diastolic_bp, glucose, cholesterol,
            creatinine, diabetes, hypertension, MAP, comorbidity_count,
            high_creatinine_flag, hyperglycemia_flag, abnormal_bp_flag,
            age_creatinine_risk, sex_Male, sex_Other,
            BMI_Category_Obese, BMI_Category_Overweight, BMI_Category_Underweight
        ]])

        # Predictions
        diagnosis_pred = diagnosis_model.predict(input_features)[0]
        severity_pred = severity_model.predict(input_features)[0]

        diagnosis_label = reverse_diagnosis_map.get(diagnosis_pred, 'Unknown')
        severity_label = reverse_severity_map.get(severity_pred, 'Unknown')

        diagnosis_result = {'disease': diagnosis_label, 'severity': severity_label}

    return render_template('Disease.html', diagnosis_result=diagnosis_result)


if __name__ == "__main__":
    app.run(debug=True)
