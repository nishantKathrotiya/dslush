# preload.py
from app.services.embedding import EmbeddingService
from app.data.loader import load_all_data
from app.services.chroma_store import add_profiles_to_chroma


def preload_data():
    print("📦 Loading datasets...")
    
    # ✅ Step 1: Load talent data
    data = load_all_data()
    talents = data["talents"].head(50).to_dict(orient="records")

    # ✅ Step 2: Flatten lists to strings (Chroma needs primitive values)
    for t in talents:
        for key in t:
            if isinstance(t[key], list):
                t[key] = ", ".join(t[key])

    # ✅ Step 3: Embed & store in Chroma
    embedder = EmbeddingService.get_instance()
    bios = [t["Bio"] for t in talents]
    bio_vectors = embedder.embed_batch(bios)
    add_profiles_to_chroma(talents, bio_vectors)
    print("✅ Preload complete.")

if __name__ == "__main__":
    preload_data()



