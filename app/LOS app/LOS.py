from flask import Flask, render_template, request
import joblib
import os
import numpy as np

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "best_los_model (1).pkl")
model = joblib.load(model_path)

# Numeric encoding dictionaries
ADMISSION_TYPE_ENCODING = {
    'URGENT': 10.50514107,
    'ELECTIVE': 11.10701614,
    'EW EMER.': 6.09199923,
    'DIRECT EMER.': 6.66631163,
    'EU OBSERVATION': 0.27959982,
    'OBSERVATION ADMIT': 7.29466506,
    'DIRECT OBSERVATION': 0.53996339,
    'AMBULATORY OBSERVATION': 0.08693663,
    'SURGICAL SAME DAY ADMISSION': 4.32125206
}

ADMISSION_LOCATION_ENCODING = {
    'TRANSFER FROM HOSPITAL': 8.3721428,
    'TRANSFER FROM SKILLED NURSING FACILITY': 10.06164297,
    'INTERNAL TRANSFER TO OR FROM PSYCH': 8.5,
    'PHYSICIAN REFERRAL': 5.12127397,
    'EMERGENCY ROOM': 5.63717044,
    'PACU': 0.39633448,
    'PROCEDURE SITE': 1.01370599,
    'WALK-IN/SELF REFERRAL': 2.17361762,
    'INFORMATION NOT AVAILABLE': 5.0,
    'CLINIC REFERRAL': 8.9251711
}

# Options
INSURANCE_OPTIONS = ['Medicaid', 'Medicare', 'Other']
LANGUAGE_OPTIONS = ['ENGLISH', 'OTHERS']
MARITAL_STATUS_OPTIONS = ['SINGLE', 'MARRIED', 'WIDOWED', 'DIVORCED']
DRG_TYPE_OPTIONS = ['HCFA', 'others']
GENDER_OPTIONS = ['male', 'female']

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Get form values
        admission_type = request.form.get("admission_type")
        admission_location = request.form.get("admission_location")
        insurance = request.form.get("insurance")
        language = request.form.get("language")
        marital_status = request.form.get("marital_status")
        drg_type = request.form.get("drg_type")
        gender = request.form.get("gender")
        age = request.form.get("age")

        # Check if all fields are filled
        if not all([admission_type, admission_location, insurance, language,
                    marital_status, drg_type, gender, age]):
            prediction = "⚠️ Please fill all fields"
        else:
            try:
                age = float(age)

                # --- Encode admission_type and location ---
                admission_type_val = ADMISSION_TYPE_ENCODING.get(admission_type, 0)
                admission_location_val = ADMISSION_LOCATION_ENCODING.get(admission_location, 0)

                # --- One-hot encode categorical features ---
                # Insurance
                insurance_Medicare = 1 if insurance == "Medicare" else 0
                insurance_Other = 1 if insurance == "Other" else 0
                # Medicaid -> both 0

                # Language
                language_OTHERS = 1 if language != "ENGLISH" else 0

                # Marital status
                marital_status_MARRIED = 1 if marital_status == "MARRIED" else 0
                marital_status_SINGLE = 1 if marital_status == "SINGLE" else 0
                marital_status_WIDOWED = 1 if marital_status == "WIDOWED" else 0
                # DIVORCED -> all 0

                # DRG type
                drg_type_HCFA = 1 if drg_type == "HCFA" else 0

                # Gender
                gender_M = 1 if gender == "male" else 0
                # female -> 0

                # --- Build input array ---
                input_features = np.array([[admission_type_val,
                                            admission_location_val,
                                            age,
                                            insurance_Medicare,
                                            insurance_Other,
                                            language_OTHERS,
                                            marital_status_MARRIED,
                                            marital_status_SINGLE,
                                            marital_status_WIDOWED,
                                            drg_type_HCFA,
                                            gender_M]])

                # --- Predict LOS ---
                predicted_los = round(float(model.predict(input_features)[0]), 2)
                prediction = f"Predicted Length of Stay: {predicted_los} days"

            except Exception as e:
                prediction = f"❌ Prediction failed: {str(e)}"

    return render_template("index.html",
                           admission_type_options=ADMISSION_TYPE_ENCODING.keys(),
                           admission_location_options=ADMISSION_LOCATION_ENCODING.keys(),
                           insurance_options=INSURANCE_OPTIONS,
                           language_options=LANGUAGE_OPTIONS,
                           marital_status_options=MARITAL_STATUS_OPTIONS,
                           drg_type_options=DRG_TYPE_OPTIONS,
                           gender_options=GENDER_OPTIONS,
                           prediction=prediction)


    
    
if __name__ == "__main__":
    app.run(debug=True, port = 5001)
