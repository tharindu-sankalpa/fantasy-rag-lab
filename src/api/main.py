# Dependencies:
# pip install fastapi uvicorn structlog

import structlog
from fastapi import FastAPI
from src.core.config import settings
from src.utils.logger import logger
from src.api.endpoints import rag_naive, rag_advanced, rag_hybrid, rag_graph, rag_agentic

def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI application.

    Returns:
        FastAPI: The configured application instance.
    """
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        openapi_url=f"{settings.API_PREFIX}/openapi.json",
        docs_url=f"{settings.API_PREFIX}/docs",
    )
    
    @app.on_event("startup")
    async def startup_event() -> None:
        """
        Actions to perform on application startup.
        """
        logger.info("application_setup", status="started", version=settings.VERSION)

    @app.get("/health")
    async def health_check() -> dict:
        """
        Health check endpoint.

        Returns:
            dict: Status of the application.
        """
        return {"status": "ok", "app": settings.PROJECT_NAME}

    app.include_router(rag_naive.router, prefix="/rag/naive", tags=["Naive RAG"])
    app.include_router(rag_advanced.router, prefix="/rag/advanced", tags=["Advanced RAG"])
    app.include_router(rag_hybrid.router, prefix="/rag/hybrid", tags=["Hybrid RAG"])
    app.include_router(rag_graph.router, prefix="/rag/graph", tags=["Graph RAG"])
    app.include_router(rag_agentic.router, prefix="/rag/agentic", tags=["Agentic RAG"])
    
    return app

app = create_app()
