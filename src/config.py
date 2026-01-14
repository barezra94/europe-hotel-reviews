from functools import lru_cache

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    vector_store_path: str = "./vector_store"
    collection_name: str = "hotel_reviews"
    data_path: str = "./data/Hotel_Reviews.csv"
    ingestion_batch_size: int = 100
    ingestion_sample_size: int = 5000


@lru_cache
def get_settings() -> Settings:
    return Settings()
