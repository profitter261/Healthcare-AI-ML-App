from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
import logging
from Chatbot.clinical_backend import ClinicalRAG  # <-- import your RAG pipeline
from flask_ngrok import run_with_ngrok

# Load environment variables
load_dotenv()

app = Flask(__name__)
run_with_ngrok(app)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load API details
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("CLINICAL_TRIAL_MODEL", "meta-llama/llama-3.1-8b-instruct")

print("Loaded model:", os.getenv("CLINICAL_TRIAL_MODEL"))
print("API key starts with:", os.getenv("OPENROUTER_API_KEY")[:10])

if not OPENROUTER_API_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY not found in .env file")

# Initialize RAG pipeline (this loads Qdrant + Embeddings)
logging.info("Initializing ClinicalRAG pipeline...")
rag = ClinicalRAG()

# ✅ Preload some dummy or real medical text chunks (replace with your preprocessed data)
dummy_chunks = [
    {"id": 1, "text": "Asthma causes airway inflammation leading to breathing difficulty.", "citation": "Medical Handbook, p. 394"},
    {"id": 2, "text": "Symptoms of asthma include wheezing, coughing, and chest tightness.", "citation": "Medical Handbook, p. 395"},
    {"id": 3, "text": "Diabetes occurs when blood sugar levels remain high due to insulin issues.", "citation": "Medical Handbook, p. 210"},
    {"id": 4, "text": "Common signs of diabetes are excessive thirst, frequent urination, and fatigue.", "citation": "Medical Handbook, p. 211"}
]
rag.ingest_data(dummy_chunks)
logging.info("✅ Dummy medical knowledge base loaded.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    """Handles patient questions and returns descriptive medical responses."""
    data = request.get_json()
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({"error": "Empty query"}), 400

    try:
        answer, citations = rag.answer_question(user_query)
        return jsonify({
            "answer": answer
        })

    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return jsonify({"error": "Something went wrong", "details": str(e)}), 500


if __name__ == "__main__":
    app.run()
