from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import urlparse
from collections import Counter
import pandas as pd
from helpers import _extract_domain

def evaluate_source_diversity(
    search_calls: List[Dict[str, Any]]
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Evaluate source/domain diversity for a list of Tavily search tool result objects.
    
    Returns
    -------
    summary_df : pd.DataFrame
        One row per Tavily call with source diversity metrics.
    details : list[dict]
        Per-call details for notebook rendering.
    """
    rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []

    for idx, call in enumerate(search_calls, start=1):
        call_id = call.get("tool_call_id")
        results = call.get("results", []) or []

        query = None
        content_obj = call.get("content")
        if isinstance(content_obj, str):
            try:
                import json
                parsed = json.loads(content_obj)
                query = parsed.get("query")
            except Exception:
                query = None

        domains = []
        invalid_url_results = []

        for r in results:
            url = r.get("url")
            domain = _extract_domain(url)
            if domain:
                domains.append(domain)
            else:
                invalid_url_results.append({
                    "title": r.get("title"),
                    "url": url,
                })

        total_sources = len(domains)
        domain_counts = Counter(domains)
        unique_sources = len(domain_counts)

        unique_source_pct = round(
            (unique_sources / total_sources) * 100, 1
        ) if total_sources > 0 else None

        most_frequent_source = None
        most_frequent_source_count = None

        if domain_counts:
            domain, count = domain_counts.most_common(1)[0]
            if count > 1:
                most_frequent_source = domain
                most_frequent_source_count = count

        # Optional simple diversity score (0-100), if you want it
        diversity_score = round(unique_source_pct, 1) if unique_source_pct is not None else None

        note = None
        if invalid_url_results:
            note = (
                f"Excluded {len(invalid_url_results)} result(s) with missing or invalid URLs."
            )
        
        if unique_source_pct is not None and unique_source_pct < 50.0:
            note = (note + " " if note else "") + "⚠ Low source diversity (less than 50% unique domains). Results may be dominated by a single source."
        elif unique_source_pct is not None and unique_source_pct >= 80.0:
            note = (note + " " if note else "") + "✅ Good source diversity (80% or more unique domains)."

        row = {
            "tool_call_index": idx,
            "tool_call_id": call_id,
            "total_sources": total_sources,
            "unique_sources": unique_sources,
            "unique_source_pct": unique_source_pct,
            "diversity_score": diversity_score,
            "most_frequent_source": most_frequent_source,
            "most_frequent_source_count": most_frequent_source_count,
            "note": note,
            "query": query,
        }

        rows.append(row)
        details.append({
            **row,
            "domain_counts": dict(domain_counts),
            "invalid_url_results": invalid_url_results,
        })

    summary_df = pd.DataFrame(rows)

    preferred_cols = [
        "tool_call_index",
        "tool_call_id",
        "total_sources",
        "unique_sources",
        "unique_source_pct",
        "diversity_score",
        "most_frequent_source",
        "most_frequent_source_count",
        "note",
        "query",
    ]
    summary_df = summary_df[[c for c in preferred_cols if c in summary_df.columns]]

    return summary_df, details

def render_source_diversity_report(summary_df: pd.DataFrame, details: List[Dict[str, Any]]) -> None:
    from IPython.display import display, HTML
    import html as html_lib

    display(summary_df)

    for d in details:
        summary_text = html_lib.escape(f"Search #{d['tool_call_index']} — {d['tool_call_id']}")

        # Build body HTML
        parts = []

        if d.get("query"):
            parts.append(f"<div><b>Query:</b> {html_lib.escape(str(d['query']))}</div>")

        # Key metrics
        usp = d.get("unique_source_pct")
        usp_text = f"{usp}%" if usp is not None else "N/A"
        parts.append(
            "<div style='margin-top:0.35rem;'>"
            f"<b>Total sources:</b> {d.get('total_sources')}<br>"
            f"<b>Unique sources:</b> {d.get('unique_sources')}<br>"
            f"<b>Unique source percentage:</b> {usp_text}"
            "</div>"
        )

        # Most frequent source
        if d.get("most_frequent_source"):
            parts.append(
                "<div style='margin-top:0.35rem;'>"
                f"<b>Most frequent source:</b> {html_lib.escape(str(d['most_frequent_source']))} "
                f"({d.get('most_frequent_source_count')} references)"
                "</div>"
            )
        else:
            parts.append(
                "<div style='margin-top:0.35rem;'><b>Most frequent source:</b> None (no repeated domains)</div>"
            )

        # Note
        if d.get("note"):
            parts.append(
                f"<div style='margin-top:0.35rem;'><b>Note:</b> {html_lib.escape(str(d['note']))}</div>"
            )

        # Domain breakdown list
        domain_counts = d.get("domain_counts") or {}
        if domain_counts:
            sorted_domains = sorted(domain_counts.items(), key=lambda x: (-x[1], x[0]))
            li = "\n".join(
                f"<li><code>{html_lib.escape(domain)}</code>: {count}</li>"
                for domain, count in sorted_domains
            )
            parts.append(
                "<div style='margin-top:0.5rem;'><b>Domain breakdown</b></div>"
                f"<ul style='margin-top:0.25rem;'>{li}</ul>"
            )

        body_html = "\n".join(parts)

        display(HTML(f"""
                        <details style="margin: 0.5rem 0 0.75rem 0;">
                        <summary style="cursor:pointer; font-weight:500; font-size:1.0rem; line-height:1.2;">
                            {summary_text}
                        </summary>
                        <div style="margin-top:0.5rem;">
                            {body_html}
                        </div>
                        </details>
                    """))