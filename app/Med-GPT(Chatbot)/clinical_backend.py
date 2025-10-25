import os
import requests
import logging
from dotenv import load_dotenv
from Chatbot.embedding_utils import EmbeddingModel
from qdrant_client import QdrantClient, models

load_dotenv()  # Load API key and model from .env

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("CLINICAL_TRIAL_MODEL", "meta-llama/llama-3.1-8b-instruct")

class VectorDB:
    def __init__(self, location=":memory:", collection_name="medical_chunks", vector_size=384):
        self.client = QdrantClient(location=location)
        self.collection_name = collection_name
        self.vector_params = models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        self._create_collection()

    def _create_collection(self):
        try:
            self.client.get_collection(collection_name=self.collection_name)
            logging.info(f"Collection '{self.collection_name}' already exists.")
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=self.vector_params
            )
            logging.info(f"Collection '{self.collection_name}' created.")

    def upsert_chunks(self, chunks, embeddings):
        points = [
            models.PointStruct(id=chunk['id'], vector=embeddings[i], payload=chunk)
            for i, chunk in enumerate(chunks)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def search_chunks(self, query_embedding, limit=5):
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )


class ClinicalRAG:
    def __init__(self):
        self.embedder = EmbeddingModel()
        dummy = self.embedder.generate_embeddings(["test"])[0]
        self.vector_db = VectorDB(vector_size=len(dummy))

    def ingest_data(self, chunks):
        embeddings = self.embedder.generate_embeddings([c["text"] for c in chunks])
        self.vector_db.upsert_chunks(chunks, embeddings)

    def call_openrouter(self, prompt):
        """Call the OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        data = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": (
                    "You are MedGPT, a professional and friendly medical assistant. Use both the provided background and your general medical knowledge to answer clearly and accurately. If additional details are relevant and commonly known in medicine, you may include them."
                    "Do not mention internal terms like 'context', 'chunks', or 'sources'. "     
                )},

                {"role": "user", "content": prompt}
            ],
            "max_tokens": 512,
            "temperature": 0.7,
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            logging.error(f"OpenRouter API Error: {response.text}")
            return "Error fetching response from OpenRouter."

    def answer_question(self, question):
        """Combine retrieved context and call OpenRouter."""
        query_embedding = self.embedder.generate_embeddings([question])[0]
        results = self.vector_db.search_chunks(query_embedding, limit=5)

        context = ""
        for i, hit in enumerate(results):
            citation = hit.payload.get("citation", "No citation available")
            context += f"--- Chunk {i+1} (Source: {citation}) ---\n{hit.payload['text']}\n\n"

        prompt = f"""
            Here is some background medical information to assist you:
            
            {context}
            
            Now, please answer the patient's question below in a natural, conversational, and accurate way:
            
            Question: {question}
            
            Your response should be clear, patient-friendly, and free of technical or internal notes.
            """


        answer = self.call_openrouter(prompt)
        citations = [hit.payload.get("citation", "No citation available") for hit in results]
        return answer, citations


if __name__ == "__main__":
    # Example Usage
    rag = ClinicalRAG()
    dummy_chunks = [
        {"id": 1, "text": "Asthma causes airway inflammation leading to difficulty in breathing.", "citation": "Medical Handbook, p. 394"},
        {"id": 2, "text": "Symptoms include wheezing, coughing, and shortness of breath.", "citation": "Medical Handbook, p. 395"},
    ]
    rag.ingest_data(dummy_chunks)

    user_query = "What are the symptoms of asthma?"
    answer, sources = rag.answer_question(user_query)

    print("Answer:", answer)
    print("Citations:", sources)
