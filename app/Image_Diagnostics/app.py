from flask import Flask, render_template_string, request, session, redirect, url_for
from tensorflow.keras.preprocessing import image
from PIL import Image
import onnxruntime as ort
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer
import os, warnings

# Ignore unnecessary TensorFlow and Keras warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 💡 IMPORTANT: Initialize Flask app and set a secret key for session management
app = Flask(__name__)
# NOTE: Change this secret key for production environments
app.secret_key = 'your_super_secret_key_change_this_in_production' 

# --- Model Paths ---
XRAY_MODEL_PATH = os.path.join(BASE_DIR, 'xray_densenet_complete_model')
MRI_MODEL_PATH = os.path.join(BASE_DIR, 'densenet121.onnx')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Globals ---
xray_model = None
mri_session = None

XRAY_CLASS_NAMES = [
    "Normal (No Finding)", "Atelectasis", "Consolidation", "Infiltration", 
    "Pneumothorax", "Edema", "Emphysema", "Fibrosis", "Effusion", 
    "Pneumonia", "Pleural_thickening", "Cardiomegaly", "Nodule Mass", "Hernia"
] 
MRI_CLASS_NAMES = [
    "AD_MildDemented", "AD_ModerateDemented", "AD_VeryMildDemented", 
    "BT_glioma", "BT_meningioma", "BT_pituitary", "MS", "Normal"
]

# --- Class Descriptions ---
XRAY_CLASS_INFO = {
    "Normal (No Finding)": "No signs of disease or abnormalities detected on the X-ray.",
    "Atelectasis": "Partial or complete collapse of a part of the lung, often due to blocked airways.",
    "Consolidation": "Lung tissue filled with liquid instead of air, often caused by pneumonia or infection.",
    "Infiltration": "Subtle increase in lung opacity, possibly indicating early infection or inflammation.",
    "Pneumothorax": "Air present in the pleural space, which can cause lung collapse.",
    "Edema": "Fluid accumulation in the lungs, commonly linked to heart conditions.",
    "Emphysema": "A chronic lung condition that causes shortness of breath due to damaged air sacs.",
    "Fibrosis": "Thickening or scarring of lung tissue that reduces elasticity and breathing efficiency.",
    "Effusion": "Fluid accumulation between the layers of tissue lining the lungs and chest cavity.",
    "Pneumonia": "Infection that inflames air sacs in one or both lungs, possibly filling them with fluid.",
    "Pleural_thickening": "Thickening of the pleural lining, often due to infection, injury, or asbestos exposure.",
    "Cardiomegaly": "Enlargement of the heart, often a sign of underlying heart disease.",
    "Nodule Mass": "A small abnormal growth in the lung that could be benign or malignant.",
    "Hernia": "Protrusion of an organ through the chest wall or diaphragm into another area."
}

MRI_CLASS_INFO = {
    "AD_MildDemented": "Early signs of Alzheimer’s disease with mild cognitive decline.",
    "AD_ModerateDemented": "More pronounced Alzheimer’s symptoms including memory loss and confusion.",
    "AD_VeryMildDemented": "Very early-stage Alzheimer’s showing minimal symptoms.",
    "BT_glioma": "A type of brain tumor originating from glial cells within the brain or spine.",
    "BT_meningioma": "A usually benign tumor arising from the meninges, the membranes around the brain and spinal cord.",
    "BT_pituitary": "Tumor located in the pituitary gland that can affect hormone production.",
    "MS": "Multiple Sclerosis — an autoimmune condition affecting the central nervous system.",
    "Normal": "No structural brain abnormalities detected on the MRI."
}

# --- Load Models (Modified to use plain text output to prevent UnicodeEncodeError) ---
def load_models():
    global xray_model, mri_session
    try:
        if os.path.isdir(XRAY_MODEL_PATH):
            xray_model = TFSMLayer(XRAY_MODEL_PATH, call_endpoint='serving_default', trainable=False)
            # Dummy call to initialize the model layers
            xray_model(np.zeros((1, 224, 224, 3), dtype=np.float32))
            print("--- X-ray model loaded.")
        else:
            print(f"!!! X-ray model not found at {XRAY_MODEL_PATH}")
    except Exception as e:
        print(f"!!! Error loading X-ray model: {e}")

    try:
        if os.path.exists(MRI_MODEL_PATH):
            mri_session = ort.InferenceSession(MRI_MODEL_PATH)
            print("--- MRI model loaded.")
        else:
            print(f"!!! MRI model not found at {MRI_MODEL_PATH}")
    except Exception as e:
        print(f"!!! Error loading MRI model: {e}")

load_models()

# --- Preprocess ---
def preprocess_image(img_path):
    # Ensure image is converted to RGB (3 channels) and resized for model input
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0).astype('float32') / 255.0
    return img_array

# --- HTML (Embedded Frontend) ---
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Image Diagnostics</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Poppins:wght@400;500;700&display=swap');

/* Root colors - Light Theme (default, but background is dark) */
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

/* --- Center Content & Form Container --- */
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
h1.page-title {
    font-size: 2.5rem;
    font-family: 'Orbitron', sans-serif;
    text-shadow: 0 0 10px var(--secondary-glow), 0 0 20px var(--glow-color);
    color: var(--secondary-glow);
    margin-bottom: 2rem;
    text-align: center;
}
body.light-mode .center-content h1 {
  text-shadow: none; /* remove the neon glow */
  background: none;  /* remove the gradient */
  -webkit-text-fill-color: var(--glow-color); /* or pick a readable solid color */
  color: var(--glow-color); /* fallback for browsers that don't support background-clip */
}

.center-content p {
    max-width: 600px;
    margin-top: 1rem;
    font-size: 1.1rem;
    opacity: 1;
    color: var(--secondary-glow);
    text-shadow: 0 0 5px rgba(255, 255, 255, 0.5);
}

.nav-panel h2 {
    color: var(--glow-color);
    text-shadow: 0 0 8px var(--glow-color);
}

h2 {
    color: var(--glow-color);
    text-align: center;
    margin-top: 2rem;
    text-shadow: 0 0 5px var(--glow-color);
}
hr {
    display: none; /* Hide the separator line */
}

/* --- Side-by-Side Layout --- */
.diagnostic-sections {
    display: flex;
    justify-content: space-between;
    gap: 30px;
    margin-top: 20px;
}
.diagnostic-section {
    flex: 1; /* Each section takes equal width */
    min-width: 0; /* Allows flex-shrink */
    padding: 20px;
    border-radius: 15px;
    background: rgba(0,0,0,0.1);
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 0 10px rgba(0,0,0,0.3);
}

/* File Input & Button Styling */
form { 
    margin: 20px 0; 
    text-align: center; 
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 15px;
}
input[type=file] { 
    padding: 12px;
    border: 2px solid var(--secondary-glow); 
    border-radius: 10px; 
    background: var(--input-bg-light); 
    color: var(--input-text-dark);
    font-weight: 500;
    width: 90%; 
    max-width: 400px;
    box-sizing: border-box;
}

/* Custom file input look for theme */
input[type="file"]::-webkit-file-upload-button {
    visibility: hidden;
}
input[type="file"]::before {
    content: 'Select Image File';
    display: inline-block;
    background: var(--glow-color);
    color: var(--input-text-dark);
    border-radius: 8px;
    padding: 8px 15px;
    outline: none;
    white-space: nowrap;
    cursor: pointer;
    font-weight: 700;
    text-shadow: 0 0 5px var(--input-bg-light);
    transition: all 0.3s;
}
input[type="file"]:hover::before {
    background: var(--secondary-glow);
    color: #000;
}
input[type="file"]:active {
    outline: 0;
}
input[type="file"]:valid {
    border-color: var(--glow-color);
}
.dark-mode input[type=file] {
    background: #1f1f1f;
    color: #eee;
}

button { 
    padding: 15px 30px; 
    font-size: 18px; 
    font-weight: bold; 
    border-radius: 15px; 
    border: 2px solid var(--secondary-glow); 
    background: rgba(0,255,200,0.15); 
    color: var(--text-color);
    cursor: pointer; 
    transition: 0.3s;
    text-transform: uppercase;
}
button:hover { 
    background: var(--secondary-glow); 
    color: #000;
    box-shadow: 0 0 40px var(--secondary-glow); 
    transform: translateY(-3px);
}

/* --- Result Card Styling --- */
.result-card { 
    margin-top: 25px; 
    padding: 20px; /* Smaller padding for side-by-side */
    border-radius: 15px; 
    background: rgba(29, 155, 240, 0.05);
    border: 2px solid var(--glow-color);
    box-shadow: 0 0 20px var(--glow-color);
    font-size: 1rem;
    font-weight: 500;
    display: none; /* Hidden by default via CSS */
}
.result-card p {
    margin: 10px 0;
}
.result-card p strong {
    color: var(--secondary-glow);
    font-weight: 700;
}
.error { color: var(--error-color); font-weight: bold; text-shadow: 0 0 5px var(--error-color); }

/* Progress Bar */
.progress { 
    height: 12px; 
    background: #333; 
    border-radius: 6px; 
    overflow: hidden; 
    margin-top: 10px; 
    box-shadow: inset 0 0 5px rgba(0,0,0,0.5);
}
.bar { 
    height: 100%; 
    background: linear-gradient(90deg, var(--glow-color), var(--secondary-glow)); 
    transition: width 1s ease-out;
    border-radius: 6px;
}

/* --- Theme Toggle --- */
/* Theme toggle */
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


/* Media query for smaller screens */
@media (max-width: 768px) {
    .nav-panel {
        width: 60%;
        min-width: unset;
        max-width: unset;
    }
    .left-toggle.active {
        left: 60%; 
    }
    .right-toggle.active {
        right: 60%;
    }
}

/* Responsive adjustments */
@media (max-width: 950px) {
    /* Stack the two diagnostic sections vertically on smaller screens */
    .diagnostic-sections {
        flex-direction: column;
        gap: 20px;
    }
}
@media (max-width: 768px) {
    .nav-panel { width: 70%; min-width: unset; max-width: unset; }
    .left-toggle.active { left: 70%; }
    .right-toggle.active { right: 70%; }
    .center-content { padding: 1rem; }
    .content-shifted { margin-left: 0; margin-right: 0;}
    h1.page-title { font-size: 2rem; }
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
    <h1 class="page-title">AI Image Diagnostics</h1>
    <div class="container">
        
        <div class="diagnostic-sections">
            
            <div class="diagnostic-section">
                <h2>X-Ray Diagnosis</h2>
                <form action="{{ url_for('predict_xray') }}" method="post" enctype="multipart/form-data">
                    <input type="file" name="xray_image" accept="image/*" required>
                    <button type="submit"><i class="bi bi-bandaid"></i> Analyze X-Ray</button>
                </form>

                {% if xray_result %}
                <div class="result-card" id="xrayResultCard" style="{% if not xray_result %}display:none{% endif %}">
                     {% if xray_result is string %}
                         <p class="error">{{ xray_result }}</p>
                     {% elif xray_result %}
                         <p><strong>Predicted Condition:</strong> {{ xray_result['prediction'] }}</p>
                         <p><strong>About the Condition:</strong> {{ xray_result['description'] }}</p>
                     {% endif %}
                </div>
                {% endif %}
            </div>

            <div class="diagnostic-section">
                <h2>MRI Diagnosis</h2>
                <form action="{{ url_for('predict_mri') }}" method="post" enctype="multipart/form-data">
                    <input type="file" name="mri_image" accept="image/*" required>
                    <button type="submit"><i class="bi bi-brain"></i> Analyze MRI</button>
                </form>

                {% if mri_result %}
                <div class="result-card" id="mriResultCard" style="{% if not mri_result %}display:none{% endif %}">
                    {% if mri_result is string %}
                        <p class="error">{{ mri_result }}</p>
                    {% elif mri_result %}
                        <p><strong>Predicted Condition:</strong> {{ mri_result['prediction'] }}</p>
                        <p><strong>About the Condition:</strong> {{ mri_result['description'] }}</p>
                    {% endif %}
                </div>
                {% endif %}
            </div>

        </div> </div>
</div>

<script>
const body = document.body;
const leftNav = document.getElementById('leftNav');
const rightNav = document.getElementById('rightNav');
const leftToggle = document.getElementById('leftToggle');
const rightToggle = document.getElementById('rightToggle');
const centerContent = document.getElementById('centerContent');
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

// --- Result Showing Logic ---
$(document).ready(function() {
    // Show X-Ray result card if it was rendered by Flask (i.e., a POST request was made)
    const xrayCard = $('#xrayResultCard');
    if (xrayCard.length) {
        xrayCard.fadeIn(500); 
    }
    // Show MRI result card if it was rendered by Flask
    const mriCard = $('#mriResultCard');
    if (mriCard.length) {
        mriCard.fadeIn(500); 
    }
});
</script>
</body>
</html>
"""

# --- Routes ---
@app.route("/")
def image_diagnostics_page():
    # Retrieve results from session
    xray_result = session.pop('xray_result', None)
    mri_result = session.pop('mri_result', None)
    # Correctly renders the HTML string defined above
    return render_template_string(HTML_PAGE, xray_result=xray_result, mri_result=mri_result)

@app.route("/predict_xray", methods=["POST"])
def predict_xray():
    # Clear any previous results
    session.pop('xray_result', None)
    session.pop('mri_result', None) 
    
    if xray_model is None:
        # FIX: Replaced emoji with safe text
        session['xray_result'] = "[ERROR] X-ray model not loaded. Check server logs."
        return redirect(url_for('image_diagnostics_page'))
        
    file = request.files.get("xray_image")
    if not file:
        session['xray_result'] = "No file uploaded."
        return redirect(url_for('image_diagnostics_page'))
    
    # Save the file to the temporary upload folder
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    try:
        img = preprocess_image(filepath)
        pred_output = xray_model(img)
        # Handle dictionary output (if model has named outputs) or direct tensor output
        pred = list(pred_output.values())[0].numpy() if isinstance(pred_output, dict) else pred_output.numpy()
        probs = tf.nn.softmax(pred).numpy().flatten()
        idx = np.argmax(probs)
        
        pred_class = XRAY_CLASS_NAMES[idx]
        result = {
            'prediction': pred_class,
            'description': XRAY_CLASS_INFO.get(pred_class, "No information available for this condition."),
        }

        session['xray_result'] = result
        
    except Exception as e:
        session['xray_result'] = f"Prediction Error: {e}"
    finally:
        # Always clean up the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
        
    # Redirect to the main page to show the results
    return redirect(url_for('image_diagnostics_page'))

@app.route("/predict_mri", methods=["POST"])
def predict_mri():
    # Clear any previous results
    session.pop('xray_result', None)
    session.pop('mri_result', None)

    if mri_session is None:
        # FIX: Replaced emoji with safe text
        session['mri_result'] = "[ERROR] MRI model not loaded. Check server logs."
        return redirect(url_for('image_diagnostics_page'))
        
    file = request.files.get("mri_image")
    if not file:
        session['mri_result'] = "No file uploaded."
        return redirect(url_for('image_diagnostics_page'))
    
    # Save the file to the temporary upload folder
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    try:
        img = preprocess_image(filepath)
        # Apply normalization and format for ONNX/PyTorch model expectation (N, C, H, W)
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = np.transpose(img, (0,3,1,2)).astype(np.float32)
        
        input_name = mri_session.get_inputs()[0].name
        outputs = mri_session.run(None, {input_name: img})
        
        pred = np.array(outputs[0]).flatten()
        idx = np.argmax(pred)
        
        # Calculate probability/confidence (assuming outputs are logits or scores)
        # Note: If this model returns a probability distribution, skip the softmax.
        confidence_score = pred[idx] / np.sum(np.exp(pred - np.max(pred))) if np.sum(np.exp(pred)) > 0 else pred[idx]

        pred_class = MRI_CLASS_NAMES[idx]
        result = {
            'prediction': pred_class,
            'description': MRI_CLASS_INFO.get(pred_class, "No information available for this condition."),
            'confidence': float(round(confidence_score * 100, 2))
        }

        session['mri_result'] = result
    except Exception as e:
        session['mri_result'] = f"Prediction Error: {e}"
    finally:
        # Always clean up the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
        
    # Redirect to the main page to show the results
    return redirect(url_for('image_diagnostics_page'))

if __name__ == "__main__":
    # FIX: The line app.add_url_rule was removed in the last step, preventing the TemplateNotFound error.
    app.run(debug=True)
