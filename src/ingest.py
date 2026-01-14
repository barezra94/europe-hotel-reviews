import ast
from datetime import datetime

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions

from config import get_settings


def parse_tags(tags_str):
    """Parse stringified list of tags from CSV."""
    try:
        return ast.literal_eval(tags_str)
    except (ValueError, SyntaxError):
        return []


def build_document(row):
    """Combine review fields into a single document string."""
    return f"Hotel: {row['Hotel_Name']} | Positive: {row['Positive_Review']} | Negative: {row['Negative_Review']}"


def build_metadata(row):
    tags = parse_tags(row["Tags"])

    days_since_review = 0
    review_date_str = row.get("Review_Date", None)
    if pd.notna(review_date_str):
        try:
            review_date = datetime.strptime(str(review_date_str), "%m/%d/%Y")
            today = datetime.now()
            days_since_review = (today - review_date).days
            days_since_review = max(0, days_since_review)
        except (ValueError, TypeError) as e:
            days_since_review = 3650

    return {
        "hotel_name": row["Hotel_Name"],
        "reviewer_nationality": row["Reviewer_Nationality"],
        "tags": ", ".join(tags),
        "days_since_review": days_since_review,
    }


def run_ingestion():
    settings = get_settings()

    print("Loading dataset...")
    df = pd.read_csv(settings.data_path)
    df = df.sample(n=settings.ingestion_sample_size, random_state=42)

    client = chromadb.PersistentClient(path=settings.vector_store_path)
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name=settings.embedding_model,
    )
    collection = client.get_or_create_collection(
        name=settings.collection_name, embedding_function=embedding_fn
    )

    print(f"Ingesting {len(df)} documents...")
    batch_size = settings.ingestion_batch_size
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]

        documents = [build_document(row) for _, row in batch.iterrows()]
        metadatas = [build_metadata(row) for _, row in batch.iterrows()]
        ids = [str(idx) for idx in batch.index]

        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print(f"  Batch {i // batch_size + 1}/{len(df) // batch_size + 1}")

    print("Done.")


if __name__ == "__main__":
    run_ingestion()
