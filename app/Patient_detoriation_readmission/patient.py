import os
import sys
import numpy as np
from flask import Flask, request, render_template_string
from tensorflow.keras.models import load_model
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Model Paths ---
# NOTE: The models 'gru_critical_forecaster.h5' and 'gru_readmission.h5' must exist 
# in the same directory as this script for the prediction logic to work.
DETERIORATION_MODEL_PATH = os.path.join(BASE_DIR, "gru_critical_forecaster.h5")
READMISSION_MODEL_PATH = os.path.join(BASE_DIR, "gru_readmission.h5")

# --- Global Models ---
deterioration_model = None
readmission_model = None
MODELS_LOADED = False

# --- Load Models ---
try:
    if not os.path.exists(DETERIORATION_MODEL_PATH) or not os.path.exists(READMISSION_MODEL_PATH):
        print("⚠️ GRU model files not found.")
    else:
        # Added compile=False for robustness during simple loading
        deterioration_model = load_model(DETERIORATION_MODEL_PATH, compile=False)
        readmission_model = load_model(READMISSION_MODEL_PATH, compile=False)
        MODELS_LOADED = True
        print("✅ GRU models loaded successfully.")
except Exception as e:
    print(f"❌ ERROR loading GRU models: {e}")

# --- Embedded HTML Template (CYBERPUNK/NEON UI with Bar Charts) ---
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Patient Risk Prediction</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Poppins:wght@400;500;700&display=swap');

:root {
    --glow-color: #1d9bf0;
    --secondary-glow: #00ffd9;
    --shadow-light: 0 0 10px rgba(29, 155, 240, 0.6), 0 0 20px rgba(29, 155, 240, 0.4);
    --text-color: #fff;
    --input-bg-dark: #1f1f1f;
    --input-bg-light: #f0f8ff; 
    --input-text-dark: #0a0a0a;
    --error-color: #ee5253;
}

.light-mode {
    background: #f4f7f6 !important;
    color: #333 !important;
    --glow-color: #007bff;
    --secondary-glow: #00a68c;
    --shadow-light: 0 5px 15px rgba(0,0,0,0.1);
    --text-color: #333;
    --input-bg-dark: #fff;
    --input-bg-light: #fff; 
    --input-text-dark: #333;
    --error-color: #c0392b;
}

body {
    margin: 0;
    font-family: 'Poppins', sans-serif;
    background: radial-gradient(circle at center, #0b0c10, #001f3f);
    color: var(--text-color);
    transition: background 0.5s, color 0.5s;
    overflow-x: hidden;
}

body.light-mode {
   background: #f4f7f6 !important;
   color: #333 !important;
   --glow-color: #007bff;
   --secondary-glow: #00a68c;
   --shadow-light: 0 5px 15px rgba(0,0,0,0.1);
   --text-color: #333;
   --input-bg-dark: #fff;
   --input-bg-light: #fff; 
   --input-text-dark: #333;
   --error-color: #c0392b;
}

body.dark-mode {
   background: radial-gradient(circle at center, #0b0c10, #001f3f);
   color: var(--text-color);
}
/* Inputs fully follow theme */
body.light-mode input[type="number"] {
    background: var(--input-bg-light);
    color: var(--input-text-dark);
    border-color: var(--secondary-glow);
}

body.dark-mode input[type="number"] {
    background: var(--input-bg-dark);
    color: #eee;
    border-color: var(--secondary-glow);
}

/* Result card follows theme */
body.light-mode #resultCard {
    background: rgba(255,255,255,0.9);
    color: #333;
    border-color: var(--glow-color);
    box-shadow: 0 0 20px rgba(0,0,0,0.2);
}

body.dark-mode #resultCard {
    background: rgba(29, 155, 240, 0.05);
    color: var(--text-color);
    border-color: var(--glow-color);
    box-shadow: 0 0 20px var(--glow-color);
}

/* Bar chart text */
body.light-mode .chart-label,
body.light-mode .status {
    color: #333;
    text-shadow: none;
}

body.light-mode .det-bar { background: linear-gradient(90deg, #007bff, #33cabb); }
body.light-mode .readm-bar { background: linear-gradient(90deg, #00a68c, #33cabb); }
body.light-mode .bar-background { background: #ddd; }

/* Navigation panels */
.nav-panel {
    position: fixed;
    top: 0;
    width: 18%;
    min-width: 180px;
    max-width: 250px;
    height: 100vh;
    background: rgba(255,255,255,0.08); 
    backdrop-filter: blur(12px);
    box-shadow: 0 0 20px rgba(0,0,0,0.6);
    transition: transform 0.5s ease;
    z-index: 10;
    padding: 2rem 1rem;
    overflow-y: auto;
}

.light-mode .nav-panel {
    background: rgba(255, 255, 255, 0.9);
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

.left-nav { left: 0; border-radius: 0 20px 20px 0; transform: translateX(0); }
.right-nav { right: 0; border-radius: 20px 0 0 20px; transform: translateX(0); }
.nav-hidden-left { transform: translateX(calc(-1 * (100% + 5px))); }
.nav-hidden-right { transform: translateX(calc(100% + 5px)); }

.nav-panel h2 { 
    color: var(--glow-color); 
    text-align: center; 
    margin-bottom: 2rem; 
    font-size: 1.5rem; 
    text-shadow: 0 0 8px var(--glow-color); 
}

.light-mode .nav-panel h2 { text-shadow: none; }

.nav-panel ul { list-style: none; padding: 0; }
.nav-panel li { margin: 1.5rem 0; text-align: center; font-weight: 500; padding: 0; cursor: pointer; transition: all 0.3s ease; }

.nav-panel li a { 
    color: var(--secondary-glow); 
    text-decoration: none; 
    display: flex; 
    align-items: center;
    justify-content: center;
    padding: 10px 0;
    text-shadow: 0 0 5px rgba(255,255,255,0.2);
}

.light-mode .nav-panel li a { 
    color: var(--secondary-glow); 
    text-shadow: none;
}

.nav-panel li a:hover { 
    color: var(--glow-color); 
    text-shadow: 0 0 10px var(--glow-color); 
    transform: scale(1.05); 
}

.light-mode .nav-panel li a:hover { text-shadow: none; color: var(--glow-color); }

.toggle-btn {
    position: fixed;
    top: 50%;
    transform: translateY(-50%);
    background: rgba(0,0,0,0.7);
    border: 2px solid var(--glow-color);
    box-shadow: var(--shadow-light);
    color: var(--glow-color);
    font-size: 1.2rem;
    padding: 0.8rem 0.6rem;
    cursor: pointer;
    z-index: 30;
    transition: all 0.5s ease;
}

.light-mode .toggle-btn {
    background: #fff;
    border: 2px solid var(--glow-color);
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}

.toggle-btn:hover { 
    background: rgba(0,0,0,1); 
    box-shadow: 0 0 15px var(--glow-color), 0 0 30px var(--glow-color);
}

.light-mode .toggle-btn:hover { background: #eee; box-shadow: 0 2px 10px var(--glow-color);}

.left-toggle { left: 0; border-radius: 0 10px 10px 0; }
.right-toggle { right: 0; border-radius: 10px 0 0 10px; }
.left-toggle.active { left: calc(18% + 5px); border-radius: 10px; } 
.right-toggle.active { right: calc(18% + 5px); border-radius: 10px; } 

/* --- Center Content & Form --- */
.center-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    margin-left: 0; 
    margin-right: 0;
    min-height: 100vh;
    padding: 2rem;
    transition: margin 0.5s ease;
}
.content-shifted {
    margin-left: calc(18% + 20px);
    margin-right: calc(18% + 20px);
}
h1.page-title {
    font-size: 2.5rem;
    font-family: 'Orbitron', sans-serif;
    text-shadow: 0 0 10px var(--secondary-glow), 0 0 20px var(--glow-color);
    color: var(--secondary-glow);
    margin-bottom: 1rem;
    text-align: center;
}
body.light-mode .center-content h1 {
  text-shadow: none; /* remove the neon glow */
  background: none;  /* remove the gradient */
  -webkit-text-fill-color: var(--glow-color); /* or pick a readable solid color */
  color: var(--glow-color); /* fallback for browsers that don't support background-clip */
}
p.subtext { text-align:center; color: #aaa; margin-bottom: 30px; }
.container {
    max-width: 900px;
    width: 100%;
    padding: 40px;
    border-radius: 25px;
    background: rgba(0,255,200,0.05); 
    border: 2px solid var(--secondary-glow);
    box-shadow: 0 0 30px rgba(0,255,200,0.3);
    transition: all 0.5s ease;
}
form { 
    display: grid; 
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
    gap: 30px; 
}
.form-group { display: flex; flex-direction: column; }
label { font-weight: 500; margin-bottom: 8px; color: var(--text-color); font-size: 0.95rem; }
input[type="number"] {
    padding: 12px 15px; 
    border-radius: 10px; 
    border: 2px solid var(--secondary-glow);
    background: var(--input-bg-light); 
    color: var(--input-text-dark);
    font-weight: 500;
    width: 100%;
    box-sizing: border-box;
    transition: all 0.3s;
}
input[type="number"]:focus {
    outline: none;
    border-color: var(--glow-color);
    box-shadow: 0 0 15px rgba(29, 155, 240, 0.4);
}
.dark-mode input[type="number"] {
    background: #1f1f1f;
    border-color: var(--secondary-glow);
    color: #eee;
}
button[type="submit"] { 
    grid-column: 1 / -1; 
    padding: 15px; 
    font-size: 18px; 
    font-weight: bold; 
    border-radius: 15px; 
    border: 2px solid var(--secondary-glow); 
    background: rgba(0,255,200,0.15); 
    color: var(--text-color);
    cursor: pointer; 
    transition: 0.3s;
}
button[type="submit"]:hover { 
    background: var(--secondary-glow); 
    color: #000;
    box-shadow: 0 0 40px var(--secondary-glow); 
    transform: translateY(-3px);
}

/* Remove number input spinners */
input[type=number]::-webkit-inner-spin-button,
input[type=number]::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }
input[type=number] { -moz-appearance: textfield; }

/* --- Result Card Styling --- */
#resultCard {
    text-align: left;
    margin-top: 40px;
    padding: 30px;
    border-radius: 15px;
    background: rgba(29, 155, 240, 0.05);
    border: 2px solid var(--glow-color);
    box-shadow: 0 0 20px var(--glow-color);
    font-size: 1.05rem;
    font-weight: 500;
    /* CRITICAL: Hide by default via CSS */
    display: none; 
}
.result-card h3 { 
    margin-top: 0;
    color: var(--secondary-glow); 
    text-shadow: 0 0 5px var(--secondary-glow);
    border-bottom: 1px dashed var(--glow-color);
    padding-bottom: 10px;
    margin-bottom: 20px;
}
.result-card p strong {
    color: var(--secondary-glow);
    font-weight: 700;
}
.status { font-weight: 700; padding: 3px 8px; border-radius: 4px; }
.low { color: var(--low-risk-color); background: rgba(16, 172, 132, 0.3); border: 1px solid var(--low-risk-color);} 
.moderate { color: #ff9f43; background: rgba(255, 159, 67, 0.3); border: 1px solid #ff9f43;}
.high { color: var(--high-risk-color); background: rgba(238, 82, 83, 0.3); border: 1px solid var(--high-risk-color);}
.error { color: #e74c3c; font-weight: bold; text-shadow: none; }

/* --- Bar Chart Styling --- */
.chart-container { margin-top: 20px; }
.chart-item { margin-bottom: 15px; }
.chart-label { font-weight: 500; font-size: 0.9rem; margin-bottom: 5px; color: #ccc; }
.bar-background {
    height: 12px;
    background: #333; /* Dark background for the bar */
    border-radius: 6px;
    overflow: hidden;
    box-shadow: inset 0 0 5px rgba(0,0,0,0.5);
}
.bar-fill {
    height: 100%;
    transition: width 1s ease-out;
    border-radius: 6px;
    box-shadow: 0 0 5px rgba(255,255,255,0.4);
}
/* Color classes for the bar fill */
.det-bar { background: linear-gradient(90deg, #1d9bf0, var(--high-risk-color)); }
.readm-bar { background: linear-gradient(90deg, #00ffd9, var(--high-risk-color)); }

/* --- Theme Toggle --- */
.theme-toggle {
    position: absolute;
    top: 1rem;
    right: 1rem;
    z-index: 40;
}
.theme-btn {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 1.8rem;
  color: var(--secondary-glow);
  transition: transform 0.3s ease, color 0.3s ease;
}

.theme-btn:hover {
  transform: rotate(20deg);
}

body.light-mode .theme-btn {
  color: var(--text-color);
}


/* Responsive adjustments */
@media (max-width: 768px) {
    .nav-panel { width: 70%; min-width: unset; max-width: unset; }
    .left-toggle.active { left: 70%; }
    .right-toggle.active { right: 70%; }
    .center-content { padding: 1rem; }
}
</style>
</head>
<body>

<div class="theme-toggle">
  <button id="themeToggle" class="theme-btn">
    <i class="bi bi-moon-fill"></i>
  </button>
</div>

<div class="nav-panel left-nav" id="leftNav">
    <h2>Navigation</h2>
    <ul>
        <li><a href="/"><i class="bi bi-house-door"></i> Home</a></li>
        <li><a href="/Disease"><i class="bi bi-heart-pulse"></i> Disease Prediction</a></li>
        <li><a href="/LOS"><i class="bi bi-graph-up"></i> LOS Prediction</a></li>
        <li><a href="/Cluster"><i class="bi bi-people"></i> Patient Cohorts</a></li>
        <li><a href="/chat"><i class="bi bi-question-circle"></i> Help</a></li>
    </ul>
</div>

<div class="nav-panel right-nav" id="rightNav">
    <h2>AI Tools</h2>
    <ul>
        <li><a href="/Patient"><i class="bi bi-lightbulb"></i> Patient Risk Assessment</a></li>
        <li><a href="/Summarizer"><i class="bi bi-file-earmark-text"></i> Notes Summarizer</a></li>
        <li><a href="/Image-Diagnostics"><i class="bi bi-image"></i> Image Diagnostics</a></li>
        <li><a href="/Senti"><i class="bi bi-chat-dots"></i> Feedback Analysis</a></li>
        <li><a href="/admin"><i class="bi bi-table"></i> Admin Dashboard</a></li>
    </ul>
</div>

<button class="toggle-btn left-toggle" id="leftToggle">☰</button>
<button class="toggle-btn right-toggle" id="rightToggle">☰</button>

<div class="center-content" id="centerContent">
    <h1 class="page-title">Patient Risk Prediction</h1>
    <p class="subtext">Predict patient deterioration & readmission risks using vital signs and clinical data.</p>

    <div class="container">
        <form action="/predict" method="post" id="predictionForm">
            <div class="form-group">
                <label for="bp_systolic">BP Systolic (mmHg)</label>
                <input type="number" step="any" name="bp_systolic" id="bp_systolic" required placeholder="e.g., 120">
            </div>
            
            <div class="form-group">
                <label for="bp_diastolic">BP Diastolic (mmHg)</label>
                <input type="number" step="any" name="bp_diastolic" id="bp_diastolic" required placeholder="e.g., 80">
            </div>
            
            <div class="form-group">
                <label for="heart_rate">Heart Rate (bpm)</label>
                <input type="number" step="any" name="heart_rate" id="heart_rate" required placeholder="e.g., 75">
            </div>
            
            <div class="form-group">
                <label for="respiratory_rate">Respiratory Rate (breaths/min)</label>
                <input type="number" step="any" name="respiratory_rate" id="respiratory_rate" required placeholder="e.g., 16">
            </div>
            
            <div class="form-group">
                <label for="temperature">Temperature (°C)</label>
                <input type="number" step="any" name="temperature" id="temperature" required placeholder="e.g., 37.0">
            </div>
            
            <div class="form-group">
                <label for="oxygen_saturation">Oxygen Saturation (%)</label>
                <input type="number" step="any" name="oxygen_saturation" id="oxygen_saturation" required placeholder="e.g., 98">
            </div>
            
            <div class="form-group">
                <label for="med_adherence">Medication Adherence (%)</label>
                <input type="number" step="any" name="med_adherence" id="med_adherence" required placeholder="e.g., 90">
            </div>
            
            <div class="form-group">
                <label for="symptom_severity">Symptom Severity (1–10)</label>
                <input type="number" step="any" name="symptom_severity" id="symptom_severity" required placeholder="e.g., 5">
            </div>
            
            <button type="submit"><i class="bi bi-activity"></i> Predict Risk</button>
        </form>

        {% if result %}
        <div class="result-card" id="resultCard">
            {% if result.error %}
                <p class="error">{{ result.error }}</p>
            {% else %}
                <h3>Prediction Results</h3>
                
                <div class="chart-container">
                    <div class="chart-item">
                        <div class="chart-label">Deterioration Risk: <strong>{{ result.det_prob }}%</strong></div>
                        <div class="bar-background">
                            <div class="bar-fill det-bar" style="width: {{ result.det_prob }}%;"></div>
                        </div>
                    </div>
                    
                    <div class="chart-item">
                        <div class="chart-label">Readmission Risk: <strong>{{ result.readm_prob }}%</strong></div>
                        <div class="bar-background">
                            <div class="bar-fill readm-bar" style="width: {{ result.readm_prob }}%;"></div>
                        </div>
                    </div>
                </div>

                <p style="margin-top: 20px;"><strong>Deterioration Status:</strong> 
                    <span class="status {{ result.det_status|lower }}">{{ result.det_status }}</span></p>
                <p><strong>Readmission Status:</strong> 
                    <span class="status {{ result.readm_status|lower }}">{{ result.readm_status }}</span></p>
                
                <p style="margin-top: 20px;"><strong>Key Action Thresholds:</strong></p>
                <ul>
                    {% if result.det_status|lower == 'high' or result.readm_status|lower == 'high' %}
                        <li style="color: var(--high-risk-color);"><i class="bi bi-exclamation-octagon-fill"></i> **CRITICAL ALERT:** Immediate clinical review is required.</li>
                    {% elif result.det_status|lower == 'moderate' or result.readm_status|lower == 'moderate' %}
                        <li style="color: #ff9f43;"><i class="bi bi-exclamation-triangle-fill"></i> **WARNING:** Implement enhanced monitoring and care planning.</li>
                    {% else %}
                        <li style="color: var(--low-risk-color);"><i class="bi bi-check-circle-fill"></i> **LOW RISK:** Continue standard care plan and monitoring.</li>
                    {% endif %}
                </ul>
            {% endif %}
        </div>
        {% endif %}
    </div>
</div>

<script>
const body = document.body;
const leftNav = document.getElementById('leftNav');
const rightNav = document.getElementById('rightNav');
const leftToggle = document.getElementById('leftToggle');
const rightToggle = document.getElementById('rightToggle');
const centerContent = document.getElementById('centerContent');
const themeToggle = document.getElementById('themeToggle');
const resultCard = document.getElementById('resultCard');
const themeToggleBtn = document.getElementById('themeToggle');
const icon = themeToggleBtn.querySelector('i');
const savedTheme = localStorage.getItem('theme');

// --- UI Logic (Navigation & Theme) ---

function updateContentMargins() {
    let marginLeft = '0';
    let marginRight = '0';
    const panelWidth = 'calc(18% + 20px)';

    if (!leftNav.classList.contains('nav-hidden-left')) {
        marginLeft = panelWidth;
    }
    if (!rightNav.classList.contains('nav-hidden-right')) {
        marginRight = panelWidth;
    }
    
    if (window.innerWidth <= 768) {
        centerContent.style.marginLeft = '0';
        centerContent.style.marginRight = '0';
    } else {
        centerContent.style.marginLeft = marginLeft;
        centerContent.style.marginRight = marginRight;
    }
}

// Navigation toggles
leftToggle.addEventListener('click', () => {
    const hidden = leftNav.classList.toggle('nav-hidden-left');
    leftToggle.classList.toggle('active', !hidden);
    updateContentMargins();
});
rightToggle.addEventListener('click', () => {
    const hidden = rightNav.classList.toggle('nav-hidden-right');
    rightToggle.classList.toggle('active', !hidden);
    updateContentMargins();
});

// Apply saved theme on load
if (savedTheme === 'light') {
  body.classList.add('light-mode');
  icon.classList.replace('bi-moon-fill', 'bi-sun-fill');
} else {
  body.classList.add('dark-mode');
  icon.classList.replace('bi-sun-fill', 'bi-moon-fill');
}

// Toggle theme on click
themeToggleBtn.addEventListener('click', () => {
  if (body.classList.contains('dark-mode')) {
    body.classList.remove('dark-mode');
    body.classList.add('light-mode');
    icon.classList.replace('bi-moon-fill', 'bi-sun-fill');
    localStorage.setItem('theme', 'light');
  } else {
    body.classList.remove('light-mode');
    body.classList.add('dark-mode');
    icon.classList.replace('bi-sun-fill', 'bi-moon-fill');
    localStorage.setItem('theme', 'dark');
  }
});


function checkScreenSize(){
    if(window.innerWidth <= 768){
        leftNav.classList.add('nav-hidden-left');
        rightNav.classList.add('nav-hidden-right');
        leftToggle.classList.remove('active');
        rightToggle.classList.remove('active');
        centerContent.style.marginLeft = '0';
        centerContent.style.marginRight = '0';
    } else {
        updateContentMargins();
    }
}
checkScreenSize();
window.addEventListener('resize', checkScreenSize);

// --- Result Hiding/Showing Logic ---

// CRITICAL: Use jQuery to show the card ONLY if it was rendered by the Flask server
$(document).ready(function() {
    const resultCardElement = $('#resultCard');
    
    // The resultCard element will only exist in the HTML on a POST request (successful prediction).
    // The CSS default is display: none; 
    if (resultCardElement.length) {
        // Use a fade-in effect to show the card nicely if it was rendered by the server
        resultCardElement.fadeIn(500); 
    }
});
</script>
</body>
</html>
"""

# --- Routes ---
@app.route('/')
def index():
    # Render the Patient Risk Assessment page by default
    return render_template_string(HTML_PAGE, result=None)

@app.route('/Patient')
def patient_page():
    # This route handles the navigation link for the main form page
    return render_template_string(HTML_PAGE, result=None)

@app.route('/predict', methods=['POST'])
def predict_risk():
    """
    Handles the form submission and performs the model prediction.
    """
    if not MODELS_LOADED:
        return render_template_string(HTML_PAGE, result={'error': "❌ Models not loaded. Check server logs and ensure required .h5 files exist."})

    data = request.form.to_dict()
    try:
        # 1. Extract and validate features
        features = [
            float(data['bp_systolic']),
            float(data['bp_diastolic']),
            float(data['heart_rate']),
            float(data['respiratory_rate']),
            float(data['temperature']),
            float(data['oxygen_saturation']),
            float(data['med_adherence']),
            float(data['symptom_severity'])
        ]
        
        # 2. Reshape input for GRU models (assuming a sequence length of 30)
        X_input = np.array([features], dtype='float32')
        X_input = np.repeat(X_input[:, np.newaxis, :], 30, axis=1)

        # 3. Predict probabilities
        # NOTE: verbose=0 suppresses Keras output
        det_prob = float(deterioration_model.predict(X_input, verbose=0)[0][0])
        readm_prob = float(readmission_model.predict(X_input, verbose=0)[0][0])

        # 4. Determine status based on thresholds
        det_status = "High" if det_prob > 0.7 else "Moderate" if det_prob > 0.4 else "Low"
        readm_status = "High" if readm_prob > 0.7 else "Moderate" if readm_prob > 0.4 else "Low"

        result = {
            'det_prob': round(det_prob * 100, 2),
            'readm_prob': round(readm_prob * 100, 2),
            'det_status': det_status,
            'readm_status': readm_status
        }
    except Exception as e:
        # Return an error message to the user if something goes wrong
        result = {'error': f"Error processing input or during prediction: {e}"}

    # Pass the result object back to the template
    return render_template_string(HTML_PAGE, result=result)

if __name__ == '__main__':
    # Running the app on the specified host and port
    app.run(host='127.0.0.1', debug=True, port=5000)