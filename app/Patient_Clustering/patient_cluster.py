from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from joblib import load
import gdown
import os
import plotly.graph_objects as go
import json
import io
import base64

app = Flask(__name__)

# ============ Utility Functions ============

# Get the absolute path of the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model_and_scaler():
    """
    Loads the trained Random Forest model, scaler, and label encoders
    from local files in a path-safe way.
    """
    model_path = os.path.join(BASE_DIR, "random_forest_cluster.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    label_encoders_path = os.path.join(BASE_DIR, "label_encoders.pkl")

    model = load(model_path)
    scaler = load(scaler_path)
    label_encoders = load(label_encoders_path)

    return model, scaler, label_encoders

# Load your models and preprocessing tools
rf_model, scaler, label_encoders = load_model_and_scaler()

# Load the dataset locally
data_path = os.path.join(BASE_DIR, "labeled_data.csv")
df_final = pd.read_csv(data_path)

x_cluster = df_final.drop(
    ['Heart_Disease', 'Diabetes', 'Other_Cancer', 'Skin_Cancer',
     'Arthritis', 'Depression', 'cluster', 'Height_(cm)', 'Weight_(kg)'], axis=1)

categorical_Binary = ['Exercise', 'Sex', 'Smoking_History']

mappings = {
    "General_Health": {"Poor": 0, "Fair": 1, "Good": 2, "Very Good": 3, "Excellent": 4},
    "Checkup": {
        "Within the past year": 4, "Within the past 2 years": 3,
        "Within the past 5 years": 2, "5 or more years ago": 1, "Never": 0
    },
    "Age_Category": {
        "18-24": 0, "25-29": 1, "30-34": 2, "35-39": 3, "40-44": 4,
        "45-49": 5, "50-54": 6, "55-59": 7, "60-64": 8, "65-69": 9,
        "70-74": 10, "75-79": 11, "80+": 12
    },
    "Diabetes": {"No": 0, "No, pre-diabetes or borderline diabetes": 1,
                 "Yes, but female told only during pregnancy": 1, "Yes": 2}
}

def preprocess_input(data):
    for col in categorical_Binary:
        data[col] = label_encoders[col].transform([data[col]])[0]

    for col in data.columns:
        if col in mappings:
            data[col] = data[col].map(mappings[col])

    data = data.drop(['Height_(cm)', 'Weight_(kg)'], axis=1)
    data = scaler.transform(data)
    feature_names = x_cluster.columns.to_list()
    return pd.DataFrame(data, columns=feature_names)

# ============ Routes ============

@app.route('/')
def home():
    return render_template('patient_clustering.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form

        height = float(form['height'])
        weight = float(form['weight'])
        bmi = weight / ((height / 100) ** 2)

        input_dict = {
            'General_Health': form['general_health'],
            'Checkup': form['checkup'],
            'Exercise': form['exercise'],
            'Sex': form['sex'],
            'Age_Category': form['age'],
            'Height_(cm)': height,
            'Weight_(kg)': weight,
            'BMI': round(bmi, 2),
            'Smoking_History': form['smoking'],
            'Alcohol_Consumption': float(form['alcohol']),
            'Fruit_Consumption': float(form['fruit']),
            'Green_Vegetables_Consumption': float(form['veg']),
            'FriedPotato_Consumption': float(form['fries'])
        }

        data = pd.DataFrame(input_dict, index=[0])
        preprocessed_data = preprocess_input(data)

        cluster_prediction = rf_model.predict(preprocessed_data)[0]

        # Cluster description dictionary
        cluster_descriptions = {
            0: "Group 1: Generally healthy individuals with active lifestyles and balanced diets.",
            1: "Group 2: Moderate risk — may have some unhealthy habits like low exercise or higher BMI.",
            2: "Group 3: Higher health risk — often associated with poor dietary habits and higher chronic condition likelihood.",
            3: "Group 4: Older population segment with increased likelihood of multiple health conditions.",
            4: "Group 5: Individuals with sedentary lifestyles and moderate obesity risk; preventive lifestyle changes recommended.",
            5: "Group 6: Younger adults with inconsistent health habits — may show early signs of risk factors like stress or irregular exercise.",
            6: "Group 7: Individuals with high BMI and unhealthy dietary patterns — elevated risk of metabolic disorders such as diabetes or hypertension.",
            7: "Group 8: People managing existing chronic conditions; continuous medical supervision and improved lifestyle adherence advised."
        }


        disease_columns = ['Heart_Disease', 'Diabetes', 'Skin_Cancer', 'Arthritis', 'Depression']
        cluster_data = df_final[df_final['cluster'] == cluster_prediction]

        disease_stats = []

        # Calculate only percentages — no ranking
        for disease in disease_columns:
            if disease == "Diabetes":
                for label_value, label_name in zip([1, 2], ["Yes", "Intermediate"]):
                    percent = cluster_data[disease].value_counts(normalize=True).get(label_value, 0) * 100
                    disease_stats.append((f"{disease} - {label_name}", percent))
            else:
                percent = cluster_data[disease].value_counts(normalize=True).get(1, 0) * 100
                disease_stats.append((disease, percent))

        # Sort by percentage descending
        disease_stats_sorted = sorted(disease_stats, key=lambda x: x[1], reverse=True)

        # Create a simple Plotly bar chart (no ranks)
        fig = go.Figure()
        for disease, yes_percent in disease_stats_sorted:
            fig.add_trace(go.Bar(
                x=[yes_percent],
                y=[disease],
                orientation='h',
                text=f"{yes_percent:.1f}%",
                textposition='auto'
            ))

        fig.update_layout(
            title=f"Disease Likelihood for Health Group {cluster_prediction+1}",
            xaxis_title="Percentage of People with Condition",
            yaxis=dict(autorange="reversed"),
            height=400 + 50 * len(disease_columns),
        )

        graph_html = fig.to_html(full_html=False)

        cluster_info = cluster_descriptions.get(int(cluster_prediction), "No description available for this group.")

        return render_template(
            'results.html',
            cluster=int(cluster_prediction + 1),
            cluster_info=cluster_info,
            graph_html=graph_html
        )

    except Exception as e:
        return jsonify({'error': str(e)})

# ============ Run App ============
if __name__ == '__main__':
    app.run(debug=True)
