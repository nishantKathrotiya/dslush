import chromadb
from typing import List, Dict

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="talent_profiles")

def generate_id(profile: Dict, index: int) -> str:
    """
    Generate a unique ID for each profile based on first name, last name, and index.
    """
    first = profile.get("First Name", "").strip().lower().replace(" ", "_")
    last = profile.get("Last Name", "").strip().lower().replace(" ", "_")
    return f"{first}_{last}_{index}"

def add_profiles_to_chroma(profiles: List[Dict], embeddings: List[List[float]]):
    all_ids = [generate_id(profile, i) for i, profile in enumerate(profiles)]

    # 🧠 Find which IDs already exist in DB
    existing = collection.get(ids=all_ids, include=[])
    existing_ids = set(existing.get("ids", []))

    # 📦 Prepare only new profiles
    new_profiles = []
    new_embeddings = []
    new_ids = []

    for i, (profile, emb, pid) in enumerate(zip(profiles, embeddings, all_ids)):
        if pid not in existing_ids:
            new_profiles.append(profile)
            new_embeddings.append(emb)
            new_ids.append(pid)

    # ✅ Add only new profiles
    if new_profiles:
        collection.add(
            documents=[p["Bio"] for p in new_profiles],
            embeddings=new_embeddings,
            metadatas=new_profiles,
            ids=new_ids
        )
        print(f"✅ Added {len(new_profiles)} new profiles to ChromaDB.")
    else:
        print("⚠️ No new profiles to add. All already exist.")

def query_similar_profiles(job_embedding: List[float], top_k: int = 10) -> List[Dict]:
    results = collection.query(
        query_embeddings=[job_embedding],
        n_results=top_k,
        include=["metadatas", "distances"]
    )

    matches = []
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        metadata["similarity"] = round(1 - distance, 4)  # 1 - distance = similarity
        matches.append(metadata)

    return matches