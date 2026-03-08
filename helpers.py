
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
from urllib.parse import urlparse
from models import TraceSummary, AIMessage, ToolCall, ToolResult
from typing import Dict, List, Any, Optional, Tuple, Union
import re

def parse_langgraph_trace(trace_data: Dict[str, Any]) -> TraceSummary:
    """
    Parse LangGraph ReAct agent trace into structured summary
    """
    messages = trace_data['messages']
    
    query = ""
    final_output = ""
    steps = []
    tool_results = []
    total_tokens = 0
    
    for msg in reversed(messages):
        if msg["type"] == 'ai' and msg["content"] and not getattr(msg, 'tool_calls', []):
            final_output = msg["content"]
            break
    
    for i, msg in enumerate(messages):
        msg_type = msg['type']
        
        if msg_type == 'human':
            query = msg['content']
        elif msg_type == 'ai':
            # Parse AI message
            ai_msg = AIMessage(
                content=msg.get('content', ''),
                tool_calls=[],
                tokens_in=msg['response_metadata']['token_usage']['prompt_tokens'],
                tokens_out=msg['response_metadata']['token_usage']['completion_tokens']
            )
            total_tokens += ai_msg.tokens_in + ai_msg.tokens_out
            
            # Extract tool calls
            for tc in msg.get('tool_calls', []):
                tool_call = ToolCall(
                    id=tc['id'],
                    name=tc['name'],
                    args=tc['args']
                )
                ai_msg.tool_calls.append(tool_call)
            
            steps.append(ai_msg)
            
        elif msg_type == 'tool':
            # Parse tool result
            content = msg['content']
            try:
                # Parse Tavily JSON response
                tavily_data = json.loads(content)
                results = tavily_data.get('results', [])
            except:
                results = []
            
            tool_result = ToolResult(
                tool_call_id=msg['tool_call_id'],
                name=msg['name'],
                content=content,
                results=results
            )
            tool_results.append(tool_result)
    
    return TraceSummary(
        query=query,
        final_output=final_output, 
        steps=steps,
        tool_results=tool_results,
        total_tokens=total_tokens,
    )

def print_trace_summary(trace: TraceSummary):
    """Pretty print the parsed trace"""
    print(f"Query: {trace.query}")
    print(f"Final Output: {trace.final_output[:150]}..." if len(trace.final_output) > 150 else f"Final Output: {trace.final_output}")
    print(f"Total tokens: {trace.total_tokens:,}")
    print("\nExecution Steps:")
    
    for i, step in enumerate(trace.steps, 1):
        print(f"\nStep {i}:")
        print(f"  Tokens: {step.tokens_in:,} in / {step.tokens_out:,} out")
        if step.content:
            print(f"  Content preview: {step.content[:100]}...")
        
        for tc in step.tool_calls:
            print(f"  → Tool: {tc.name} ({tc.id[:8]})")
            print(f"     Args: {json.dumps(tc.args, indent=2)}")

def _extract_domain(url: Optional[str]) -> Optional[str]:
    """
    Extract normalized domain from a URL.
    """
    if not url:
        return None

    try:
        netloc = urlparse(url).netloc.lower().strip()
        if not netloc:
            return None

        # Remove common www prefix only
        if netloc.startswith("www."):
            netloc = netloc[4:]

        return netloc
    except Exception:
        return None


def _clean_llm_json(raw: str) -> str:
    """
    Removes markdown code fences like ```json ... ``` if present.
    """
    raw = raw.strip()

    # Remove opening fence like ```json or ```
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)

    # Remove closing fence
    raw = re.sub(r"\s*```$", "", raw)

    return raw.strip()


def _normalize_timeframe_to_days(timeframe: Union[str, int, float]) -> int:
    if isinstance(timeframe, (int, float)):
        return max(1, int(timeframe))

    tf = str(timeframe).strip().lower()

    aliases = {
        "day": 1,
        "today": 1,
        "week": 7,
        "month": 30,
        "quarter": 90,
        "year": 365,
    }
    if tf in aliases:
        return aliases[tf]

    parts = tf.replace("-", " ").split()

    if len(parts) == 1:
        token = parts[0]
        if token.endswith("d") and token[:-1].isdigit():
            return int(token[:-1])
        if token.endswith("w") and token[:-1].isdigit():
            return int(token[:-1]) * 7
        if token.endswith("m") and token[:-1].isdigit():
            return int(token[:-1]) * 30
        if token.endswith("y") and token[:-1].isdigit():
            return int(token[:-1]) * 365

    if len(parts) >= 2 and parts[0].isdigit():
        qty = int(parts[0])
        unit = parts[1].rstrip("s")
        unit_map = {
            "day": 1,
            "week": 7,
            "month": 30,
            "year": 365,
        }
        if unit in unit_map:
            return qty * unit_map[unit]

    raise ValueError(f"Unsupported timeframe: {timeframe}")

def _parse_published_date(date_str: str) -> datetime:
    dt = parsedate_to_datetime(date_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
