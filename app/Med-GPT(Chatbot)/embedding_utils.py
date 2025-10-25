# embedding_utils.py
import torch
from transformers import AutoModel, AutoTokenizer

class EmbeddingModel:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # Set device to GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


    def generate_embeddings(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            model_output = self.model(**inputs)
        # Mean pooling to get sentence embeddings
        sentence_embeddings = model_output.last_hidden_state.mean(dim=1)
        return sentence_embeddings.tolist()

if __name__ == '__main__':
    # Example usage
    embedding_model = EmbeddingModel()
    texts = ["This is an example sentence.", "Each sentence gets its own embedding."]
    embeddings = embedding_model.generate_embeddings(texts)
    print(f"Generated embeddings for {len(texts)} texts.")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"First embedding starts with: {embeddings[0][:5]}...")