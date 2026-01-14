# Review Intelligence Assistant (Hotel CX)

A Retrieval-Augmented Generation (RAG) service designed to help Customer Experience (CX) agents query guest feedback in natural language. This system uses `gpt-4o-mini` for generation and `ChromaDB` for semantic search, grounded in the "515K Hotel Reviews" dataset.

## 📋 Environment Assumptions & Prerequisites

* **OS:** macOS / Linux / Windows (WSL2 recommended)
* **Python:** Version 3.10 or higher
* **Memory:** Minimum 8GB RAM (Required for local vector ingestion of the sampled dataset)
* **API Keys:** Valid OpenAI API Key with access to `gpt-4o-mini` and `text-embedding-3-small` models.

## Setup Instructions

### 1. Clone & Environment
```bash
git clone git@github.com:barezra94/europe-hotel-reviews.git
cd europe-hotel-reviews

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Create a .env file in the root directory to store the OpenAI API key:
```bash
# Create a .env file to store the OPENAI_API_KEY
touch .env
```

```plaintext
OPENAI_API_KEY=your-actual-key-here
```

Some more optional variables:
```plaintext
INGESTION_BATCH_SIZE=200
INGESTION_SAMPLE_SIZE=10000
DATA_PATH=./data/other_reviews.csv
```

### 3. Data Preparation
Download the [515K Hotel Reviews Data in Europe](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe?resource=download) dataset from Kaggle.

Place the `Hotel_Reviews.csv` file inside the `data/` directory.
    Path: `./data/Hotel_Reviews.csv`

### 4. Ingestion (ETL)
Run the ingestion script to process the raw CSV, generate embeddings, and populate the local vector store.
Note: By default, this script samples 5,000 reviews to ensure reasonable interactive performance on a laptop.

```bash
python src/ingest.py
```

## Running the Service
Start the FastAPI server using Uvicorn:

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000.

### Interactive Documentation (Swagger UI)
Open your browser to http://localhost:8000/docs to test endpoints interactively.

## API Usage

### Query Reviews
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How is the breakfast?"}'
```

### Query with Hotel Filter
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Is the wifi good?", "hotel_filter": "Hotel Arena"}'
```

### Response Format
```json
{
  "answer": "Based on reviews, the breakfast at...",
  "sources": ["[Reviewer from: UK] Great breakfast buffet..."],
  "relevant": true
}
```

| Field | Description |
|-------|-------------|
| `answer` | The generated response grounded in retrieved reviews |
| `sources` | List of review excerpts used to generate the answer |
| `relevant` | Whether the retrieved context was deemed relevant to the query |

## Project Structure
```
├── src/
│   ├── app.py        # FastAPI application & endpoints
│   ├── config.py     # Settings and environment configuration
│   ├── ingest.py     # ETL script for populating the vector store
│   └── rag.py        # Core RAG pipeline (retrieval, grading, generation)
├── tests/
│   ├── golden_set.json  # Test cases for LLM-as-a-Judge evaluation
│   └── test_eval.py     # Automated quality tests
├── data/             # Hotel reviews CSV (not committed)
└── vector_store/     # ChromaDB persistence (not committed)
```

## Architecture

For detailed architecture decisions, system design, scaling strategies, and failure modes, see [DESIGN.md](./DESIGN.md).

## Running Tests
The project includes automated tests for the API endpoints and retrieval logic. Tests use an "LLM-as-a-Judge" approach to evaluate answer quality against a golden set of test cases.

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest -v
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ValidationError: openai_api_key` | Ensure `.env` file exists with a valid `OPENAI_API_KEY` |
| `Collection not found` | Run `python src/ingest.py` to populate the vector store first |
| Memory errors during ingestion | Reduce `INGESTION_SAMPLE_SIZE` in `.env` (e.g., 2000) |
| Slow response times | Ensure you're not running with the full 515K dataset; use sampling |

## License

MIT License
