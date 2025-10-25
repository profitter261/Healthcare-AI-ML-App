from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
import os
import re
from datetime import datetime
from src.llama_model import LlamaResearchAssistant
from werkzeug.utils import secure_filename
import tempfile
from gtts import gTTS

load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
llama_assistant = LlamaResearchAssistant(openrouter_api_key)

# ✅ Define expected section names from your model
SECTIONS = [
    "Key Findings",
    "Critical Observations",
    "Recommended Actions",
    "Potential Implications"
]


def clean_markdown(text: str):
    """Remove markdown formatting like ** and *"""
    text = re.sub(r"\*{1,2}", "", text)  # remove * and **
    return text.strip()


def split_into_sections(text: str):
    """
    Extract structured sections and convert content into plain lines (no asterisks)
    """
    cleaned = clean_markdown(text)

    section_map = {}
    current_section = "Key Findings"
    section_map[current_section] = []

    # Split sentences by . and newlines
    lines = re.split(r"\.\s+|\n", cleaned)

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect section headings dynamically
        matched_section = next((sec for sec in SECTIONS if sec.lower() in line.lower()), None)
        if matched_section:
            current_section = matched_section
            if current_section not in section_map:
                section_map[current_section] = []
            continue

        # ✅ Just append line as plain text (no '* ')
        section_map[current_section].append(line)

    return section_map


def normalize_summary(summary):
    """Ensure output always has clean bullet-sections"""
    if isinstance(summary, str):
        summary = split_into_sections(summary)
    else:
        formatted = {}
        for key, value in summary.items():
            cleaned_key = key.strip().title()
            if isinstance(value, str):
                formatted.update(split_into_sections(value))
            else:
                formatted[cleaned_key] = value
        summary = formatted

    summary["Generated At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return summary


@app.route("/")
def home():
    return render_template("summarizer.html")


@app.route("/summarize", methods=["POST"])
def summarize_text():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify(error="Text cannot be empty"), 400

    try:
        summary = llama_assistant.summarize_clinical_report(text)
        summary = normalize_summary(summary)

        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify(error=f"Analysis failed: {str(e)}"), 500


@app.route("/upload", methods=["POST"])
def summarize_file():
    if "file" not in request.files:
        return jsonify(error="No file uploaded"), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)

    if filename == "":
        return jsonify(error="Invalid file name"), 400

    temp_path = os.path.join(tempfile.gettempdir(), filename)
    file.save(temp_path)

    try:
        with open(temp_path, "rb") as f:
            summary = llama_assistant.summarize_clinical_report(f)

        summary = normalize_summary(summary)

        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify(error=f"File processing failed: {str(e)}"), 500


@app.route("/tts", methods=["POST"])
def tts():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify(error="No text to convert"), 400

    try:
        tts_obj = gTTS(text=text, lang="en")
        temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tts_obj.save(temp_audio.name)

        return send_file(temp_audio.name, mimetype="audio/mp3", as_attachment=False)

    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(debug=True)
