# 🧠 DataSlush Talent Matcher – AI-Powered Recommendation System

This project is a **production-ready FastAPI backend** that powers AI-based talent recommendations using:

- ✨ Hugging Face Sentence Transformers
- 🧠 ChromaDB for vector search
- 🧰 Precomputed embeddings for efficiency
- ⚡ FastAPI for API exposure

---

## 📦 Features

- Embed talent bios, skills, software using Sentence Transformers
- Store profiles in ChromaDB (vector DB)
- Recommend top-k matches from a job description
- Support for weighted scoring (bio, skill, software, location)
- Optimized for **speed** (preprocessing on container start)
- Query via API with filters like gender, location, weights

---

## 📁 Folder Structure

dataslush-backend/
├── app/
│ ├── controllers/ # FastAPI endpoints
│ ├── data/ # Load and clean CSV data
│ ├── models/ # Pydantic schemas
│ ├── services/ # Core logic (embedding, Chroma, ranking)
│ ├── tests/ # Unit test suite
│ └── main.py # FastAPI server entrypoint
├── preload.py # Preprocess and store embeddings on startup
├── requirements.txt
└── Dockerfile

---

## 🚀 Getting Started

### 🔧 Local Setup

1. **Clone the repo**:

```bash
git clone <repo-url>
cd dataslush-backend
```

2. **Create Virtual Environemnt**:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```


3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Preload ChromaDB and Run Server:**:

```bash
python preload.py
uvicorn app.main:app --reload
```






### 🔧 Docker

1. **Build Docker image:**:

```bash
docker build -t dataslush-backend .
```

2. **Run the container:**:

```bash
docker run -p 8000:8000 dataslush-backend
```


3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Preload ChromaDB and Run Server:**:

```bash
python preload.py
uvicorn app.main:app --reload
```