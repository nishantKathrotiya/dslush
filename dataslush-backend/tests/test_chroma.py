from app.services.embedding import EmbeddingService
from app.data.loader import load_all_data
from app.services.chroma_store import add_profiles_to_chroma, query_similar_profiles

# Load & embed data
data = load_all_data()
talents = data["talents"].head(50).to_dict(orient="records")

# ✅ Flatten metadata for Chroma
for t in talents:
    for key in t:
        if isinstance(t[key], list):  # Flatten lists for Chroma metadata
            t[key] = ", ".join(t[key])

# Embed
embedder = EmbeddingService.get_instance()
bios = [t["Bio"] for t in talents]
bio_vectors = embedder.embed_batch(bios)

# Store in Chroma
add_profiles_to_chroma(talents, bio_vectors)

# Query
job_text = "Looking for a creative video editor with storytelling and 2D animation skills"
job_vector = embedder.embed_single(job_text)

results = query_similar_profiles(job_vector)
for r in results:
    print(f"{r['First Name']} {r['Last Name']} → similarity: {r['similarity']}")
