import json
import logging
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class RAGPipeline:
    """
    Encapsulates the RAG pipeline with injected dependencies.

    Usage:
        # Default (uses environment config)
        rag = RAGPipeline.from_settings()

        # Custom (for testing or different configs)
        rag = RAGPipeline(client=mock_client, collection=mock_collection, model="gpt-4o")
    """

    client: OpenAI
    collection: chromadb.Collection
    model: str

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "RAGPipeline":
        """Factory method that builds the pipeline from settings."""
        settings = settings or get_settings()

        client = OpenAI(api_key=settings.openai_api_key)
        chroma_client = chromadb.PersistentClient(path=settings.vector_store_path)
        embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key,
            model_name=settings.embedding_model,
        )
        collection = chroma_client.get_collection(
            settings.collection_name, embedding_function=embedding_fn
        )

        return cls(client=client, collection=collection, model=settings.openai_model)

    def retrieve_documents(
        self, query: str, hotel_filter: str | None = None, k: int = 5
    ) -> list[str]:
        """Retrieves documents using Hybrid Search (Vector + Metadata)."""
        where_clause = {"hotel_name": hotel_filter} if hotel_filter else {}
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where_clause,
            include=["documents", "metadatas"],
        )
        # Combine document text with metadata for richer context
        enriched_docs = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            metadata_str = f"[Reviewer from: {meta.get('reviewer_nationality', 'Unknown').strip()}]"
            enriched_docs.append(f"{metadata_str} {doc}")
        return enriched_docs

    def check_relevance(self, query: str, context: str) -> bool:
        """The 'Gatekeeper': Returns True if context answers the query."""
        system_prompt = (
            "You are a RAG evaluator. Return JSON with 'is_relevant' (bool). "
            "Return true if the context contains ANY information that could help answer "
            "the user's question, even partially. Only return false if the context is "
            "completely unrelated to the question."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Q: {query}\nContext: {context}"},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return json.loads(response.choices[0].message.content).get(
                "is_relevant", True
            )
        except Exception as e:
            logger.warning("Relevance check failed: %s", e, exc_info=True)
            return True

    def generate_answer(self, query: str, context: str) -> str:
        """Generates the final grounded answer."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer using ONLY the context. When referencing reviews, "
                        "mention the hotel name(s) being discussed."
                    ),
                },
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"},
            ],
        )
        return response.choices[0].message.content

    def query(self, question: str, hotel_filter: str | None = None) -> dict:
        """Full RAG pipeline: retrieve → check → generate."""
        docs = self.retrieve_documents(question, hotel_filter)
        if not docs:
            return {"answer": "No documents found.", "sources": [], "relevant": False}

        context = "\n\n".join(docs)

        if not self.check_relevance(question, context):
            return {
                "answer": "Found documents but they don't answer your question.",
                "sources": [],
                "relevant": False,
            }

        answer = self.generate_answer(question, context)
        return {"answer": answer, "sources": docs, "relevant": True}
