import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Metrics:
    request_count: int = 0
    error_count: int = 0
    latency_samples: List[float] = field(default_factory=list)
    retrieval_failures: int = 0
    total_tokens_input: int = 0
    total_tokens_output: int = 0

    def record_request(
        self, latency: float, tokens_input: int = 0, tokens_output: int = 0
    ):
        self.request_count += 1
        self.latency_samples.append(latency)
        self.total_tokens_input += tokens_input
        self.total_tokens_output += tokens_output

        if len(self.latency_samples) > 1000:
            self.latency_samples = self.latency_samples[-1000:]

    def record_error(self):
        self.error_count += 1

    def record_retrieval_failure(self):
        self.retrieval_failures += 1

    def get_p95_latency(self) -> float:
        if not self.latency_samples:
            return 0.0
        sorted_samples = sorted(self.latency_samples)
        p95_index = int(len(sorted_samples) * 0.95)
        return (
            sorted_samples[p95_index]
            if p95_index < len(sorted_samples)
            else sorted_samples[-1]
        )

    def get_retrieval_failure_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.retrieval_failures / self.request_count

    def get_cost_per_query(self) -> float:
        if self.request_count == 0:
            return 0.0
        # Pricing Assumption - using gpt 4o-mini: $0.15 per 1M input tokens, $0.60 per 1M output tokens
        input_cost = (self.total_tokens_input / 1_000_000) * 0.15
        output_cost = (self.total_tokens_output / 1_000_000) * 0.60
        total_cost = input_cost + output_cost
        return total_cost / self.request_count

    def get_summary(self) -> Dict[str, float]:
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "p95_latency_ms": self.get_p95_latency() * 1000,
            "retrieval_failure_rate": self.get_retrieval_failure_rate(),
            "cost_per_query_usd": self.get_cost_per_query(),
        }


_metrics = Metrics()


def get_metrics() -> Metrics:
    return _metrics
