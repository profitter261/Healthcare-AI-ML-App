import zipfile, os, textwrap

# Define project structure
base_dir = "/mnt/data/healthcare_app"
src_dir = os.path.join(base_dir, "src")
os.makedirs(src_dir, exist_ok=True)

# New llama_model.py (JSON-first version skeleton)
llama_model_code = textwrap.dedent("""
import os
import json
import logging
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import pdfplumber

class LlamaResearchAssistant:
    def __init__(self, api_key=None):
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.api_base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.logger = logging.getLogger(__name__)
        self.current_model = "meta-llama/llama-3.1-8b-instruct"

    def _generate(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.current_model,
            "messages": [
                {"role": "system", "content": "Return only valid JSON with medical sections."},
                {"role": "user", "content": prompt}
            ]
        }
        res = requests.post(self.api_base_url, headers=headers, json=payload)
        return res.json()["choices"][0]["message"]["content"]

    def summarize_clinical_report(self, text: str):
        prompt = f\"""Summarize the medical report into JSON with:
        keyFindings, criticalObservations, recommendedActions, potentialImplications.\"""
        return self._generate(prompt)
""")

# Minimal functional Flask app using JSON-only assistant
app_code = textwrap.dedent("""
from flask import Flask, request, jsonify
from src.llama_model import LlamaResearchAssistant

app = Flask(__name__)
assistant = LlamaResearchAssistant()

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get("text", "")
    response = assistant.summarize_clinical_report(text)
    return jsonify({"summary": response})

if __name__ == "__main__":
    app.run(debug=True)
""")

with open(os.path.join(src_dir, "llama_model.py"), "w") as f:
    f.write(llama_model_code)

with open(os.path.join(base_dir, "app1.py"), "w") as f:
    f.write(app_code)

# Create ZIP
zip_path = "/mnt/data/medical_ai_app.zip"
with zipfile.ZipFile(zip_path, 'w') as zipf:
    for root, _, files in os.walk(base_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, base_dir)
            zipf.write(file_path, arcname)

zip_path
