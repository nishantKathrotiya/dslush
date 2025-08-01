from sentence_transformers import SentenceTransformer
from typing import List
import threading

class EmbeddingService:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        print("🔁 Loading SentenceTransformer model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # Good balance of speed + quality
        print("✅ Model Loaded.")

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = EmbeddingService()
            return cls._instance

    def embed_single(self, text: str) -> List[float]:
        return self.model.encode(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, batch_size=32, show_progress_bar=True)
