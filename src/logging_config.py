import logging
import sys
from logging import Formatter


def setup_logging():
    class StructuredFormatter(Formatter):
        def format(self, record):
            log_data = {
                "timestamp": self.formatTime(record, self.datefmt),
                "level": record.levelname,
                "message": record.getMessage(),
            }
            if hasattr(record, "request_id"):
                log_data["request_id"] = record.request_id
            if hasattr(record, "query_text"):
                log_data["query_text"] = record.query_text
            if hasattr(record, "hotel_filter"):
                log_data["hotel_filter"] = record.hotel_filter
            if hasattr(record, "retrieved_doc_count"):
                log_data["retrieved_doc_count"] = record.retrieved_doc_count
            if hasattr(record, "grader_decision"):
                log_data["grader_decision"] = record.grader_decision
            if hasattr(record, "final_latency"):
                log_data["final_latency"] = record.final_latency
            if hasattr(record, "error"):
                log_data["error"] = record.error

            parts = [f"{k}={v}" for k, v in log_data.items()]
            return " | ".join(parts)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
