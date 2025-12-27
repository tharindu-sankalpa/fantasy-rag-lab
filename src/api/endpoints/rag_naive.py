from fastapi import APIRouter
import structlog

router = APIRouter()
logger = structlog.get_logger()

@router.post("/query")
async def query_naive(q: str):
    logger.info("Naive RAG query received", query=q)
    return {"response": "Not implemented yet"}
