from flask import Flask, request, jsonify, render_template_string
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os, re
import pdfplumber, pytesseract
from pdf2image import convert_from_bytes
from textblob import TextBlob
from io import BytesIO

# --- Flask app ---
app = Flask(__name__)

# --- Model Path ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "clinical_bart_model")

# --- Load model ---
print("Loading model from:", MODEL_PATH)
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    print("Model loaded successfully.")
except Exception as e:
    print(f"!!! Error loading summarization model: {e}. Summarization functionality will be limited.")
    tokenizer = None
    model = None
    summarizer = lambda text, **kwargs: [{"summary_text": f"DUMMY SUMMARY: Model failed to load ({text[:50]}...)"}] if text else [{"summary_text": "DUMMY SUMMARY: No input text."}]


# --- Helpers ---
def clean_text(text):
    return " ".join(text.split())

def clean_pdf_text(text):
    text = re.sub(r'\(cid:\d+\)', '', text)
    return clean_text(text)

def spell_correct(text):
    return str(TextBlob(text).correct())

def extract_pdf_text(file_bytes):
    """Try pdfplumber first; fallback to OCR if empty."""
    text = ""
    file_bytes.seek(0)
    with pdfplumber.open(file_bytes) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    text = clean_pdf_text(text)

    if not text.strip():
        file_bytes.seek(0)
        images = convert_from_bytes(file_bytes.read())
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img) + " "
        text = clean_text(text)
    return text

# --- Embedded HTML Frontend ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Patient Notes Summarizer</title>
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

.light-mode body {
    background: #f4f7f6 !important;
    color: #333 !important;
}

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

.light-mode .container {
    background: #fff;
    border: 1px solid #ddd;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

h1.page-title {
    font-size: 2.5rem;
    font-family: 'Orbitron', sans-serif;
    text-shadow: 0 0 10px var(--secondary-glow), 0 0 20px var(--glow-color);
    color: var(--secondary-glow);
    margin-bottom: 2rem;
    text-align: center;
}

.light-mode h1.page-title {
    text-shadow: none;
    color: var(--glow-color);
}

h3 {
    color: var(--glow-color);
    text-align: center;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
    text-shadow: 0 0 5px var(--glow-color);
    font-size: 1.25rem;
}

.light-mode h3 {
    text-shadow: none;
    color: var(--secondary-glow);
}

.input-sections {
    display: flex;
    gap: 25px;
    margin-top: 20px;
    flex-wrap: wrap;
    align-items: flex-start;
    justify-content: center;
}

.input-section {
    flex: 1;
    min-width: 300px;
    max-width: 400px;
    padding: 20px;
    border-radius: 15px;
    background: rgba(0,0,0,0.1);
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 0 10px rgba(0,0,0,0.3);
}

.light-mode .input-section {
    background: #f9f9f9;
    border: 1px solid #ddd;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

form { 
    margin: 0; 
    text-align: center; 
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 15px;
}

textarea { 
    width: 100%; 
    height: 160px; 
    padding: 15px; 
    border-radius: 10px; 
    border: 2px solid var(--secondary-glow); 
    background: var(--input-bg-dark);
    color: var(--text-color);
    font-family: 'Poppins', sans-serif;
    box-sizing: border-box;
    transition: all 0.3s;
}

.light-mode textarea {
    background: var(--input-bg-light); 
    color: var(--input-text-dark);
}

textarea:focus {
    outline: none;
    box-shadow: 0 0 15px var(--glow-color);
}

.file-input-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 160px;
    border: 2px solid var(--secondary-glow);
    border-radius: 10px;
    background: var(--input-bg-dark);
    cursor: pointer;
    transition: all 0.3s;
    position: relative;
    overflow: hidden;
}

.file-input-wrapper:hover {
    background: rgba(0, 255, 217, 0.1);
    box-shadow: 0 0 15px var(--glow-color);
}

.light-mode .file-input-wrapper {
    background: var(--input-bg-light);
}

.light-mode .file-input-wrapper:hover {
    background: rgba(0, 255, 217, 0.05);
}

.file-input-wrapper input[type=file] {
    position: absolute;
    width: 100%;
    height: 100%;
    opacity: 0;
    cursor: pointer;
}

.file-input-label {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
    cursor: pointer;
    pointer-events: none;
}

.file-upload-icon {
    font-size: 2rem;
    color: var(--secondary-glow);
    margin-bottom: 8px;
}

.file-upload-text {
    color: var(--text-color);
    font-weight: 500;
    font-family: 'Poppins', sans-serif;
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

.light-mode button {
    background: var(--glow-color);
    color: #fff;
    border-color: var(--glow-color);
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}

button:hover { 
    background: var(--secondary-glow); 
    color: #000;
    box-shadow: 0 0 40px var(--secondary-glow); 
    transform: translateY(-3px);
}

.light-mode button:hover {
    background: var(--secondary-glow);
    color: #333;
    box-shadow: none;
}

.result-card { 
    margin-top: 30px; 
    padding: 25px; 
    border-radius: 15px; 
    background: rgba(29, 155, 240, 0.05);
    border: 2px solid var(--glow-color);
    box-shadow: 0 0 20px var(--glow-color);
    font-size: 1rem;
    font-weight: 500;
    text-align: left;
    min-height: 100px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
}

.light-mode .result-card {
    background: #e9f5ff;
    border: 1px solid var(--glow-color);
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.result-card p {
    margin: 5px 0;
    line-height: 1.6;
}

.result-card p strong {
    color: var(--secondary-glow);
    font-weight: 700;
    display: block;
    margin-bottom: 10px;
    font-size: 1.1rem;
}

.error { 
    color: var(--error-color); 
    font-weight: bold; 
    text-shadow: 0 0 5px var(--error-color);
    margin: auto;
}

.light-mode .error { text-shadow: none; }

.theme-toggle { position: fixed; top: 1rem; right: 1rem; z-index: 40; }
.theme-btn {
  background: none;
  border: 2px solid var(--glow-color);
  border-radius: 50%;
  cursor: pointer;
  font-size: 1.8rem;
  color: var(--glow-color);
  transition: transform 0.3s ease;
  padding: 8px;
  width: 50px;
  height: 50px;
  display: flex;
  align-items: center;
  justify-content: center;
  line-height: 1;
}

.theme-btn:hover {
  transform: rotate(20deg);
}

body.light-mode .theme-btn {
  color: #FF9800;
  border-color: #FF9800;
}

.spinner {
    border: 4px solid rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    border-top: 4px solid var(--glow-color);
    width: 30px;
    height: 30px;
    animation: spin 1s linear infinite;
    margin: 20px auto;
    display: none;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.light-mode .spinner {
    border: 4px solid rgba(0, 0, 0, 0.1);
    border-top: 4px solid var(--glow-color);
}

@media (max-width: 950px) {
    .input-sections {
        flex-direction: column;
        gap: 15px;
        align-items: stretch;
    }
    .input-section {
        max-width: 100%;
    }
}

@media (max-width: 768px) {
    .nav-panel { width: 70%; min-width: unset; max-width: unset; }
    .left-toggle.active { left: 70%; }
    .right-toggle.active { right: 70%; }
    .center-content { padding: 1rem; }
    h1.page-title { font-size: 2rem; }
}
</style>
</head>
<body class="dark-mode">

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
    <h1 class="page-title">Clinical Notes Summarizer</h1>
    <div class="container">
        <div class="input-sections">
            <div class="input-section">
                <h3>Enter Notes Manually</h3>
                <form id="textForm">
                    <textarea id="text" name="text" placeholder="Paste patient notes, discharge summary, or clinical reports here..."></textarea>
                    <button type="submit"><i class="bi bi-chat-left-text"></i> Generate Summary</button>
                </form>
            </div>

            <div class="input-section">
                <h3>Or Upload a File</h3>
                <form id="fileForm" enctype="multipart/form-data">
                    <div class="file-input-wrapper">
                        <label for="file" class="file-input-label">
                            <i class="bi bi-cloud-arrow-up file-upload-icon"></i>
                            <span class="file-upload-text">Upload</span>
                        </label>
                        <input type="file" id="file" name="file" accept=".pdf,.txt">
                    </div>
                    <button type="submit"><i class="bi bi-file-earmark-arrow-up"></i> Summarize File</button>
                </form>
            </div>
        </div>
        
        <div class="result-card" id="result">
            <p>Your AI-generated summary will appear here after processing.</p>
        </div>
    </div> 
</div>

<script>
const body = document.body;
const leftNav = document.getElementById('leftNav');
const rightNav = document.getElementById('rightNav');
const leftToggle = document.getElementById('leftToggle');
const rightToggle = document.getElementById('rightToggle');
const centerContent = document.getElementById('centerContent');
const resultDiv = document.getElementById("result");
const spinner = document.createElement('div');
const themeToggleBtn = document.getElementById('themeToggle');
const icon = themeToggleBtn.querySelector('i');
spinner.className = 'spinner';
resultDiv.appendChild(spinner);

function initializeTheme() {
    const savedTheme = 'dark';
    if (savedTheme === 'light') {
        body.classList.remove('dark-mode');
        body.classList.add('light-mode');
    } else {
        body.classList.add('dark-mode');
        body.classList.remove('light-mode');
    }
}

function updateContentMargins() {
    let marginLeft = '0';
    let marginRight = '0';
    const panelWidth = 'calc(18% + 20px)';
    const isLeftOpen = !leftNav.classList.contains('nav-hidden-left');
    const isRightOpen = !rightNav.classList.contains('nav-hidden-right');
    if (isLeftOpen) marginLeft = panelWidth;
    if (isRightOpen) marginRight = panelWidth;
    if (window.innerWidth <= 768) {
        centerContent.style.marginLeft = '0';
        centerContent.style.marginRight = '0';
    } else {
        centerContent.style.marginLeft = marginLeft;
        centerContent.style.marginRight = marginRight;
    }
}

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

body.classList.add('dark-mode');
icon.classList.add('bi-moon-fill');

themeToggleBtn.addEventListener('click', () => {
  if (body.classList.contains('dark-mode')) {
    body.classList.remove('dark-mode');
    body.classList.add('light-mode');
    icon.classList.replace('bi-moon-fill', 'bi-sun-fill');
  } else {
    body.classList.remove('light-mode');
    body.classList.add('dark-mode');
    icon.classList.replace('bi-sun-fill', 'bi-moon-fill');
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

function displayResult(data) {
    spinner.style.display = 'none';
    if (data.summary) {
        resultDiv.innerHTML = `<p><strong>AI-Generated Summary:</strong></p><p>${data.summary}</p>`;
    } else if (data.error) {
        resultDiv.innerHTML = `<p class="error"><i class="bi bi-exclamation-triangle-fill"></i> Error: ${data.error}</p>`;
    } else {
        resultDiv.innerHTML = `<p class="error"><i class="bi bi-exclamation-triangle-fill"></i> An unknown error occurred.</p>`;
    }
}

const textForm = document.getElementById("textForm");
textForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    resultDiv.innerHTML = '';
    spinner.style.display = 'block';
    const text = document.getElementById("text").value;
    if (!text.trim()) {
        displayResult({ error: "Please enter some text to summarize." });
        return;
    }
    try {
        const response = await fetch("./summarize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });
        const data = await response.json();
        displayResult(data);
    } catch (error) {
        displayResult({ error: "Network error or server connection failed." });
    }
});

const fileForm = document.getElementById("fileForm");
fileForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    resultDiv.innerHTML = '';
    spinner.style.display = 'block';
    const fileInput = document.getElementById("file");
    if(fileInput.files.length === 0){
        displayResult({ error: "Please select a PDF or TXT file to upload." });
        return;
    }
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    try {
        const response = await fetch("./upload", { method: "POST", body: formData });
        const data = await response.json();
        displayResult(data);
    } catch (error) {
        displayResult({ error: "Network error or file upload failed." });
    }
});

$(document).ready(function() {
    initializeTheme();
    checkScreenSize();
});

window.addEventListener('resize', checkScreenSize);
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json(force=True)
    text = clean_text(data.get("text", ""))
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        text = spell_correct(text)
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]["summary_text"]
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    try:
        print(f"Uploading file: {file.filename}")
        file_bytes = BytesIO(file.read())
        
        if file.filename.endswith(".pdf"):
            print("Extracting PDF text...")
            text = extract_pdf_text(file_bytes)
        else:
            print("Processing text file...")
            text = clean_text(file_bytes.getvalue().decode("utf-8"))
        
        print(f"Extracted text length: {len(text)}")
        if not text.strip():
            return jsonify({"error": "File contains no readable text. Try a different PDF or ensure it's not image-based."}), 400
        
        print("Correcting spelling...")
        text = spell_correct(text)
        print("Generating summary...")
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]["summary_text"]
        return jsonify({"summary": summary})
    except Exception as e:
        error_msg = f"Upload error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)