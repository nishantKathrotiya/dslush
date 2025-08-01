from app.services.embedding import EmbeddingService

text = "Video editor with 3 years of experience in vlogs and lifestyle content."

embedder = EmbeddingService.get_instance()

vector = embedder.embed_single(text)
print("Embedding length:", len(vector))
print("Sample vector values:", vector[:5])
