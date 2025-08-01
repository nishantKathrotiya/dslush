from app.services.recommender import rank_profiles_weighted , rank_profiles
from app.services.embedding import EmbeddingService
from app.data.loader import load_all_data
from app.services.chroma_store import add_profiles_to_chroma, query_similar_profiles
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
# job_text = "Looking for a podcast editor who knows Adobe Audition and FL Studio. Good with storytelling and sound design."

job_text = """
Looking for talented Video Editor with experience in Adobe Premiere Pro who can edit content in
Entertainment/Lifestyle & Vlogs categories. Historically, his team has preferred to hire
talent in Asia for cheaper but is open to all locations for this role. Content form: Any
short-form and long-forms Their top required skills: Splice & Dice, Rough Cut &
Sequencing, 2D Animation Their budget: $2500/month
"""
top_profiles = rank_profiles(job_text)

print("\nTop Weighted Recommendations:\n")
for p in top_profiles:
    print(f"{p['First Name']} {p['Last Name']} → Final: {p['final_score']}")
    


# You can adjust weights here dynamically
weights = {
    "bio": 0.5,
    "skills": 0.2,
    "software": 0.2,
    "location": 0.1
}

top_profiles = rank_profiles_weighted(job_text, top_k=5, weights=weights)

print("\nTop Weighted Recommendations:\n")
for p in top_profiles:
    print(f"{p['First Name']} {p['Last Name']} → Final: {p['final_score']} | "
          f"Bio: {p['bio_score']} | Skills: {p['skill_score']} | "
          f"Software: {p['software_score']} | Loc: {p['location_score']}")
