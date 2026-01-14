from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from src.rag import RAGPipeline

app = FastAPI(title="Hotel Review Intelligence Assistant")


class QueryRequest(BaseModel):
    query: str
    hotel_filter: str | None = None


@lru_cache
def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline.from_settings()


RAGDep = Annotated[RAGPipeline, Depends(get_rag_pipeline)]


@app.post("/query")
async def query_endpoint(request: QueryRequest, rag: RAGDep):
    return rag.query(request.query, request.hotel_filter)
