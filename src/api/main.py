from fastapi import FastAPI
from src.core.config import settings
from src.core.logging import configure_logging
import structlog

from src.api.endpoints import rag_naive, rag_advanced, rag_hybrid, rag_graph, rag_agentic

logger = structlog.get_logger()

def create_app() -> FastAPI:
    configure_logging()
    
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        openapi_url=f"{settings.API_PREFIX}/openapi.json",
        docs_url=f"{settings.API_PREFIX}/docs",
    )
    
    @app.on_event("startup")
    async def startup_event():
        logger.info("Application setup", status="started")

    @app.get("/health")
    async def health_check():
        return {"status": "ok", "app": settings.PROJECT_NAME}

    app.include_router(rag_naive.router, prefix="/rag/naive", tags=["Naive RAG"])
    app.include_router(rag_advanced.router, prefix="/rag/advanced", tags=["Advanced RAG"])
    app.include_router(rag_hybrid.router, prefix="/rag/hybrid", tags=["Hybrid RAG"])
    app.include_router(rag_graph.router, prefix="/rag/graph", tags=["Graph RAG"])
    app.include_router(rag_agentic.router, prefix="/rag/agentic", tags=["Agentic RAG"])
    
    return app

app = create_app()
