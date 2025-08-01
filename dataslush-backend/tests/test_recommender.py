from app.services.embedding import EmbeddingService
from app.data.loader import load_all_data
from app.services.chroma_store import add_profiles_to_chroma, query_similar_profiles
from app.services.recommender import rank_profiles

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

# ✅ Step 4: Run recommender
job_text = "Looking for a podcast editor who knows Adobe Audition and FL Studio. Good with storytelling and sound design."
top_profiles = rank_profiles(job_text)

# ✅ Step 5: Print Results
print("\nTop Recommendations:\n")
for p in top_profiles:
    print(f"{p['First Name']} {p['Last Name']} → Score: {p['final_score']} | Similarity: {p['similarity']}")
