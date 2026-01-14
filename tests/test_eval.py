import json
from pathlib import Path

import pytest
from openai import OpenAI

from src.config import get_settings
from src.rag import RAGPipeline

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)


def load_golden_set():
    path = Path(__file__).parent / "golden_set.json"
    with open(path) as f:
        return json.load(f)


def llm_judge(question: str, answer: str, expected_concept: str) -> str:
    """Use an LLM to evaluate if the answer addresses the question and mentions the expected concept."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a QA evaluator. Check if the answer helps the user "
                    "and mentions the expected concept. Return ONLY 'PASS' or 'FAIL'."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\nAnswer: {answer}\nExpected concept: {expected_concept}",
            },
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


@pytest.fixture(scope="module")
def pipeline():
    return RAGPipeline.from_settings()


@pytest.mark.parametrize("test_case", load_golden_set())
def test_rag_quality(test_case, pipeline):
    """Run the full RAG pipeline and grade output quality using LLM-as-a-Judge."""
    question = test_case["question"]
    hotel_filter = test_case.get("hotel_filter")

    docs = pipeline.retrieve_documents(question, hotel_filter)
    context = "\n\n".join(docs)

    if pipeline.check_relevance(question, context):
        answer = pipeline.generate_answer(question, context)
    else:
        answer = "FALLBACK: Irrelevant context."

    print(f"\nQ: {question}\nA: {answer}")

    if "FALLBACK" not in answer:
        assert docs, "Retrieval returned no documents"

    grade = llm_judge(question, answer, test_case["expected_concept"])
    assert grade == "PASS", f"LLM Judge failed. Answer: {answer}"
