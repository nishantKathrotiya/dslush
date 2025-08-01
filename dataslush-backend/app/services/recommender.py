from app.services.embedding import EmbeddingService
from app.services.chroma_store import query_similar_profiles
from app.data.loader import load_all_data
from sentence_transformers.util import cos_sim

def rank_profiles(job_description: str, top_k: int = 10) -> list:
    """
    Given a job description, return top-k most relevant talent profiles.
    """

    print('Request Came')
    embedder = EmbeddingService.get_instance()
    job_vector = embedder.embed_single(job_description)

    # Step 1: Chroma Similarity
    similar_profiles = query_similar_profiles(job_vector, top_k=50)

    # Step 2: Score with other features
    scored = []
    job_keywords = job_description.lower()

    print(f"🔎 Job Description: {job_description}")
    print(f"🔢 Total similar profiles from Chroma: {len(similar_profiles)}")

    for profile in similar_profiles:
        # Skill match score (naive text overlap)
        skills = profile.get("Skills", "").lower()
        # skill_match = sum(1 for kw in job_keywords.split() if kw in skills)

        job_words = set(job_keywords.split())
        # skills_words = set(skills.split(", "))
        # skill_match = len(job_words.intersection(skills_words))

        skill_match = sum(1 for word in job_words if word in skills)

        # Final Score
        score = profile["similarity"] * 0.7 + skill_match * 0.3

        profile["final_score"] = round(score, 4)
        print(f"→ {profile['First Name']} {profile['Last Name']} | Sim: {profile['similarity']}, Skills match: {skill_match}, Final: {profile['final_score']}")
        scored.append(profile)

    # Sort & return top-K
    return sorted(scored, key=lambda x: x["final_score"], reverse=True)[:top_k]


def rank_profiles_weighted(job_description: str, top_k=10, weights=None) -> list:
    """
    Weighted recommender using Chroma + Skill + Software + Location.
    Weights can be adjusted as a dict.
    """
    # Default weights (sum ≤ 1)
    weights = weights or {
        "bio": 0.5,        # Chroma similarity (bio)
        "skills": 0.2,     # Skill cosine similarity
        "software": 0.2,   # Software cosine similarity
        "location": 0.1    # Location match boost
    }

    # Load data for lookup
    data = load_all_data()
    talents = data["talents"]

    # Embed job description
    embedder = EmbeddingService.get_instance()
    job_vector = embedder.embed_single(job_description.lower())

    # Query top 50 from Chroma by bio similarity
    similar_profiles = query_similar_profiles(job_vector, top_k=50)
    scored_profiles = []

    for profile in similar_profiles:
        # ---------- 1. Chroma similarity ----------
        bio_sim = profile["similarity"]

        # ---------- 2. Skills cosine similarity ----------
        skills_text = profile.get("Skills", "").lower()
        skill_vector = embedder.embed_single(skills_text)
        skill_sim = cos_sim(job_vector, skill_vector).item()

        # ---------- 3. Software cosine similarity ----------
        software_text = profile.get("Software", "").lower()
        software_vector = embedder.embed_single(software_text)
        software_sim = cos_sim(job_vector, software_vector).item()

        # ---------- 4. Location match ----------
        location_match = 0
        job_lower = job_description.lower()
        city = profile.get("City", "").lower()
        country = profile.get("Country", "").lower()

        if city and city in job_lower:
            location_match = 1
        elif country and country in job_lower:
            location_match = 1

        # ---------- Weighted Score ----------
        final_score = (
            bio_sim * weights["bio"] +
            skill_sim * weights["skills"] +
            software_sim * weights["software"] +
            location_match * weights["location"]
        )

        profile["bio_score"] = round(bio_sim, 4)
        profile["skill_score"] = round(skill_sim, 4)
        profile["software_score"] = round(software_sim, 4)
        profile["location_score"] = location_match
        profile["final_score"] = round(final_score, 4)

        scored_profiles.append(profile)

    # Sort by final score and return top_k
    return sorted(scored_profiles, key=lambda x: x["final_score"], reverse=True)[:top_k]