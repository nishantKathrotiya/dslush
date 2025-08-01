from fastapi import APIRouter
from app.models.schemas import BasicRecommendRequest, WeightedRecommendRequest
from app.services.recommender import rank_profiles, rank_profiles_weighted

router = APIRouter(prefix="/recommend", tags=["Recommendation"])


@router.post("/")
def basic_recommend(request: BasicRecommendRequest):
    results = rank_profiles(request.job_description, top_k=request.top_k)
    return {
        "strategy": "basic",
        "top_k": request.top_k,
        "results": results
    }


@router.post("/weighted")
def weighted_recommend(request: WeightedRecommendRequest):
    results = rank_profiles_weighted(
        job_description=request.job_description,
        top_k=request.top_k,
        weights=request.weights
    )
    return {
        "strategy": "weighted",
        "top_k": request.top_k,
        "weights": request.weights,
        "results": results
    }
