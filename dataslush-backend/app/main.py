import utils.transport
import utils.session
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.controllers import recommendation
from app.startup.load_profiles import preload_profiles_to_chroma

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up... loading ChromaDB profiles")
    preload_profiles_to_chroma()  # ⬅️ load on startup
    yield
    print("🛑 Shutting down...")  # ⬅️ Optional cleanup logic

app = FastAPI(
    title="DataSlush AI Recommender API",
    lifespan=lifespan
)

app.include_router(recommendation.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to DataSlush API"}
