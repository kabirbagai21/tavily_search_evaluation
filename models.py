from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class ToolCall:
    id: str
    name: str
    args: Dict[str, Any]

@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    content: str
    results: List[Dict[str, Any]]

@dataclass
class AIMessage:
    content: str
    tool_calls: List[ToolCall]
    tokens_in: int
    tokens_out: int

@dataclass
class TraceSummary:
    query: str
    final_output: str
    steps: List[AIMessage]
    tool_results: List[ToolResult]
    total_tokens: int

@dataclass
class FreshnessScore:
    score: float  # 0-1 (1=perfectly fresh)
    days_old: str  # str for N/ADit
    percent_recent: float  # % of results < timeframe days old
    max_days_old: str  # str for N/A
    has_published_dates: bool
    warning: str


@dataclass
class CoverageResult:
    facets: List[str]
    facet_assignments: List[Dict[str, Any]]  # per-result facets + metadata
    facet_counts: Dict[str, int]
