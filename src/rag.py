import json
import logging
import time
import uuid
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

from src.config import Settings, get_settings
from src.metrics import get_metrics

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
    recency_weight: float = 0.3

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

        return cls(
            client=client,
            collection=collection,
            model=settings.openai_model,
            recency_weight=settings.recency_weight,
        )

    def retrieve_documents(
        self,
        query: str,
        hotel_filter: str | None = None,
        k: int = 5,
        recency_weight: float | None = None,
    ) -> list[str]:
        """
        Retrieves documents using Hybrid Search (Vector + Metadata) with recency weighting.

        Args:
            query: Search query
            hotel_filter: Optional hotel name filter
            k: Number of documents to return
            recency_weight: Weight for recency (0.0 = no recency, 1.0 = only recency)
                          If None, uses instance default (30% recency, 70% semantic similarity)
        """
        try:
            if recency_weight is None:
                recency_weight = self.recency_weight

            where_clause = {"hotel_name": hotel_filter} if hotel_filter else {}

            # Retrieve more candidates to allow re-ranking by recency
            # For hotels with many reviews, we want to prioritize newer ones
            candidate_multiplier = (
                3 if hotel_filter else 2
            )  # More candidates when filtering by hotel
            n_candidates = max(k * candidate_multiplier, 15)  # At least 15 candidates

            results = self.collection.query(
                query_texts=[query],
                n_results=n_candidates,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )

            if not results["documents"][0]:
                return []

            # Re-rank by combining semantic similarity with recency
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            # Normalize distances to similarity scores (lower distance = higher similarity)
            max_distance = max(distances) if distances else 1.0
            min_distance = min(distances) if distances else 0.0
            distance_range = (
                max_distance - min_distance if max_distance > min_distance else 1.0
            )

            scored_docs = []
            for doc, meta, dist in zip(documents, metadatas, distances):
                # Normalize semantic similarity (0 to 1, where 1 is most similar)
                similarity_score = (
                    1.0 - ((dist - min_distance) / distance_range)
                    if distance_range > 0
                    else 0.5
                )

                # Calculate recency score (newer reviews = higher score)
                days_since = meta.get("days_since_review", 0)
                if isinstance(days_since, str):
                    try:
                        days_since = int(float(days_since))
                    except (ValueError, TypeError):
                        days_since = 365  # Default to old if can't parse

                # Exponential decay: reviews from 0-30 days = 1.0, 30-90 = 0.8, 90-180 = 0.6, etc.
                # This ensures newer reviews are strongly preferred
                if days_since <= 30:
                    recency_score = 1.0
                elif days_since <= 90:
                    recency_score = 0.8
                elif days_since <= 180:
                    recency_score = 0.6
                elif days_since <= 365:
                    recency_score = 0.4
                else:
                    recency_score = 0.2

                # Combined score: weighted average of similarity and recency
                combined_score = (
                    1 - recency_weight
                ) * similarity_score + recency_weight * recency_score

                scored_docs.append((combined_score, doc, meta))

            # Sort by combined score (highest first) and take top k
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            top_docs = scored_docs[:k]

            # Combine document text with metadata for richer context
            enriched_docs = []
            for _, doc, meta in top_docs:
                days_since = meta.get("days_since_review", "Unknown")
                reviewer = meta.get("reviewer_nationality", "Unknown").strip()
                metadata_str = f"[Reviewer from: {reviewer}, {days_since} days ago]"
                enriched_docs.append(f"{metadata_str} {doc}")

            return enriched_docs
        except Exception as e:
            logger.error("Document retrieval failed: %s", e, exc_info=True)
            raise RuntimeError(f"Failed to retrieve documents: {str(e)}") from e

    def check_relevance(self, query: str, context: str) -> bool:
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
                timeout=30,
            )
            return json.loads(response.choices[0].message.content).get(
                "is_relevant", True
            )
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse relevance check JSON: %s", e, exc_info=True)
            return True
        except Exception as e:
            logger.warning("Relevance check failed: %s", e, exc_info=True)
            return True

    def generate_answer(self, query: str, context: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer using ONLY the context. When referencing reviews, "
                            "mention the hotel name(s) being discussed. "
                            "Prioritize information from more recent reviews when available, "
                            "as hotel conditions and services may have changed over time."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Context: {context}\n\nQuestion: {query}",
                    },
                ],
                timeout=30,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Answer generation failed: %s", e, exc_info=True)
            raise RuntimeError(f"Failed to generate answer: {str(e)}") from e

    def query(self, question: str, hotel_filter: str | None = None) -> dict:
        request_id = str(uuid.uuid4())
        start_time = time.time()
        metrics = get_metrics()

        try:
            logger.info(
                "Request started",
                extra={
                    "request_id": request_id,
                    "query_text": question,
                    "hotel_filter": hotel_filter,
                },
            )

            docs = self.retrieve_documents(question, hotel_filter)
            if not docs:
                latency = time.time() - start_time
                metrics.record_retrieval_failure()
                metrics.record_request(latency, 0, 0)
                logger.info(
                    "No documents retrieved",
                    extra={"request_id": request_id, "retrieved_doc_count": 0},
                )
                return {
                    "answer": "No documents found.",
                    "sources": [],
                    "relevant": False,
                }

            context = "\n\n".join(docs)
            is_relevant = self.check_relevance(question, context)

            if not is_relevant:
                latency = time.time() - start_time
                metrics.record_retrieval_failure()
                # Estimate token usage for relevance check
                tokens_input = len(question) + len(context) // 4
                metrics.record_request(latency, tokens_input, 0)
                logger.info(
                    "Retrieval failed relevance check",
                    extra={
                        "request_id": request_id,
                        "retrieved_doc_count": len(docs),
                        "grader_decision": False,
                    },
                )
                return {
                    "answer": "Found documents but they don't answer your question.",
                    "sources": [],
                    "relevant": False,
                }

            answer = self.generate_answer(question, context)
            latency = time.time() - start_time

            # Estimate token usage (rough approximation: ~4 chars per token)
            tokens_input = len(question) + len(context) // 4
            tokens_output = len(answer) // 4

            metrics.record_request(latency, tokens_input, tokens_output)

            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "retrieved_doc_count": len(docs),
                    "grader_decision": True,
                    "final_latency": latency,
                },
            )

            return {"answer": answer, "sources": docs, "relevant": True}

        except Exception as e:
            metrics.record_error()
            latency = time.time() - start_time
            metrics.record_request(latency, 0, 0)
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "final_latency": latency,
                },
                exc_info=True,
            )
            raise
