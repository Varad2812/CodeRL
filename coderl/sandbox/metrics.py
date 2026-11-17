"""Prometheus metrics for the sandbox (exposed by whatever process serves /metrics later)."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

EXECUTIONS = Counter(
    "coderl_sandbox_executions_total",
    "Sandbox executions by backend and final status.",
    ["backend", "status"],
)
TEST_CASES = Counter(
    "coderl_sandbox_test_cases_total",
    "Individual test cases executed, by outcome.",
    ["backend", "outcome"],
)
INTEGRITY_FLAGS = Counter(
    "coderl_sandbox_integrity_flags_total",
    "Executions carrying an integrity flag (possible reward hacking).",
    ["backend", "flag"],
)
LATENCY = Histogram(
    "coderl_sandbox_execution_seconds",
    "End-to-end latency of one sandbox execution (host side).",
    ["backend"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
INFLIGHT = Gauge("coderl_sandbox_inflight", "Executions currently running.", ["backend"])
CONTAINER_RECYCLES = Counter(
    "coderl_sandbox_container_recycles_total",
    "Docker pool containers recycled, by reason.",
    ["reason"],
)
