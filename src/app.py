from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.logging_config import setup_logging
from src.metrics import get_metrics
from src.rag import RAGPipeline

# Setup structured logging
setup_logging()

app = FastAPI(title="Hotel Review Intelligence Assistant")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the project root directory
BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"

# Serve static files (UI)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class QueryRequest(BaseModel):
    query: str
    hotel_filter: str | None = None


@lru_cache
def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline.from_settings()


RAGDep = Annotated[RAGPipeline, Depends(get_rag_pipeline)]


@app.post("/query")
async def query_endpoint(request: QueryRequest, rag: RAGDep):
    """Query the RAG system with natural language questions."""
    try:
        return rag.query(request.query, request.hotel_filter)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/")
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    else:
        raise HTTPException(
            status_code=404,
            detail="UI files not found. Make sure static/index.html exists.",
        )


@app.get("/metrics")
async def metrics_endpoint():
    """Get current metrics summary."""
    return get_metrics().get_summary()


@app.get("/metrics-page")
async def metrics_page():
    """Serve the metrics UI page."""
    metrics_path = STATIC_DIR / "metrics.html"
    if metrics_path.exists():
        return FileResponse(str(metrics_path))
    else:
        raise HTTPException(status_code=404, detail="Metrics page not found.")
