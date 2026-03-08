from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional, Union
import pandas as pd
from helpers import _normalize_timeframe_to_days, _parse_published_date


def evaluate_tavily_freshness(
    tavily_calls: List[Any],
    timeframe: Union[str, int, float],
    reference_time: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Evaluate freshness for each tavily_search tool call in a parsed LangGraph trace.

    - Score using only results that have published_date.
    - Skip only if all results are missing published_date.
    - Add a note whenever some results are missing dates.
    """
    window_days = _normalize_timeframe_to_days(timeframe)
    now = (reference_time or datetime.now(timezone.utc)).astimezone(timezone.utc)

    rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []

    for idx, call in enumerate(tavily_calls, start=1):
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

        if query is None:
            query = call.get("query", "")

        if not results:
            row = {
                "tool_call_index": idx,
                "tool_call_id": call_id,
                "status": "no_results",
                "timeframe_days": window_days,
                "freshness_score": None,
                "results_count": 0,
                "dated_results_count": 0,
                "within_window": 0,
                "missing_dates": 0,
                "oldest_title": None,
                "oldest_date": None,
                "newest_title": None,
                "newest_date": None,
                "note": "No search results returned.",
                "query": query,
            }
            rows.append(row)
            details.append(row.copy())
            continue

        dated_results_raw = [r for r in results if r.get("published_date")]
        missing_date_results = [r for r in results if not r.get("published_date")]

        if not dated_results_raw:
            row = {
                "tool_call_index": idx,
                "tool_call_id": call_id,
                "status": "skipped_all_dates_missing",
                "timeframe_days": window_days,
                "freshness_score": None,
                "results_count": len(results),
                "dated_results_count": 0,
                "within_window": None,
                "missing_dates": len(missing_date_results),
                "oldest_title": None,
                "oldest_date": None,
                "newest_title": None,
                "newest_date": None,
                "note": (
                    f"Skipped freshness eval: all {len(results)} results are missing "
                    "published_date."
                ),
                "query": query,
            }
            rows.append(row)
            details.append({
                **row,
                "missing_date_results": [
                    {"title": r.get("title"), "url": r.get("url")}
                    for r in missing_date_results
                ],
            })
            continue

        parsed_results = []
        for r in dated_results_raw:
            dt = _parse_published_date(r["published_date"])
            age_days = (now - dt).total_seconds() / 86400.0
            within_window = age_days <= window_days

            parsed_results.append({
                "title": r.get("title"),
                "url": r.get("url"),
                "published_date": r.get("published_date"),
                "published_dt": dt,
                "age_days": age_days,
                "within_window": within_window,
                "score": r.get("score"),
            })

        within_count = sum(1 for r in parsed_results if r["within_window"])
        coverage_ratio = within_count / len(parsed_results)

        recency_values = [
            max(0.0, 1.0 - (r["age_days"] / window_days))
            for r in parsed_results
        ]
        avg_recency = sum(recency_values) / len(recency_values)

        freshness_score = round(100 * (0.7 * coverage_ratio + 0.3 * avg_recency), 1)

        oldest = min(parsed_results, key=lambda x: x["published_dt"])
        newest = max(parsed_results, key=lambda x: x["published_dt"])

        note = None
        if missing_date_results:
            note = (
                f"Computed on {len(parsed_results)}/{len(results)} results; "
                f"{len(missing_date_results)} missing published_date."
            )

        row = {
            "tool_call_index": idx,
            "tool_call_id": call_id,
            "status": "scored_partial" if missing_date_results else "scored",
            "timeframe_days": window_days,
            "freshness_score": freshness_score,
            "results_count": len(results),
            "dated_results_count": len(parsed_results),
            "within_window": within_count,
            "missing_dates": len(missing_date_results),
            "oldest_title": oldest["title"],
            "oldest_date": oldest["published_dt"].strftime("%Y-%m-%d"),
            "newest_title": newest["title"],
            "newest_date": newest["published_dt"].strftime("%Y-%m-%d"),
            "note": note,
            "query": query,
        }

        rows.append(row)
        details.append({
            **row,
            "coverage_ratio": coverage_ratio,
            "avg_recency": avg_recency,
            "oldest_result": oldest,
            "newest_result": newest,
            "parsed_results": parsed_results,
            "missing_date_results": [
                {"title": r.get("title"), "url": r.get("url")}
                for r in missing_date_results
            ],
        })

    summary_df = pd.DataFrame(rows)

    preferred_cols = [
        "tool_call_index",
        "tool_call_id",
        "status",
        "freshness_score",
        "timeframe_days",
        "results_count",
        "dated_results_count",
        "within_window",
        "missing_dates",
        "oldest_date",
        "oldest_title",
        "newest_date",
        "newest_title",
        "note",
        "query",
    ]
    summary_df = summary_df[[c for c in preferred_cols if c in summary_df.columns]]

    return summary_df, details
def render_freshness_report(summary_df: pd.DataFrame, details: List[Dict[str, Any]]) -> None:
    from IPython.display import display, HTML
    import html as html_lib

    display(summary_df)

    for d in details:
        summary_text = html_lib.escape(f"Search #{d['tool_call_index']} — {d['tool_call_id']}")
        parts = []

        parts.append(f"<div><b>Status:</b> {html_lib.escape(str(d['status']))}</div>")
        parts.append(f"<div style='margin-top:0.35rem;'><b>Query:</b> {html_lib.escape(str(d.get('query') or '(unavailable)'))}</div>")

        if d.get("note"):
            parts.append(
                f"<div style='margin-top:0.35rem;'><b>Note:</b> {html_lib.escape(str(d['note']))}</div>"
            )

        if d["status"] not in {"no_results", "skipped_all_dates_missing"}:
            parts.append(
                "<div style='margin-top:0.5rem;'>"
                f"<b>Freshness score:</b> {d['freshness_score']}/100<br>"
                f"<b>Results within window:</b> {d['within_window']}/{d['dated_results_count']} dated results<br>"
                f"<b>Total results returned:</b> {d['results_count']}<br>"
                f"<b>Timeframe:</b> {d['timeframe_days']} days"
                "</div>"
            )

            oldest = d["oldest_result"]
            newest = d["newest_result"]

            oldest_title = html_lib.escape(str(oldest["title"]))
            oldest_url = html_lib.escape(str(oldest["url"]))
            oldest_pub = oldest["published_dt"].strftime("%Y-%m-%d %H:%M UTC")
            oldest_age = f"{oldest['age_days']:.1f} days"

            parts.append(
                "<div style='margin-top:0.75rem;'><b>Oldest dated page</b></div>"
                "<ul style='margin-top:0.25rem; padding-left:1.2rem;'>"
                f"<li>{oldest_title}</li>"
                f"<li>Published: {oldest_pub}</li>"
                f"<li>Age: {oldest_age}</li>"
                f"<li>URL: <a href=\"{oldest_url}\" target=\"_blank\" rel=\"noopener noreferrer\">{oldest_url}</a></li>"
                "</ul>"
            )

            newest_title = html_lib.escape(str(newest["title"]))
            newest_url = html_lib.escape(str(newest["url"]))
            newest_pub = newest["published_dt"].strftime("%Y-%m-%d %H:%M UTC")
            newest_age = f"{newest['age_days']:.1f} days"

            parts.append(
                "<div style='margin-top:0.75rem;'><b>Newest dated page</b></div>"
                "<ul style='margin-top:0.25rem; padding-left:1.2rem;'>"
                f"<li>{newest_title}</li>"
                f"<li>Published: {newest_pub}</li>"
                f"<li>Age: {newest_age}</li>"
                f"<li>URL: <a href=\"{newest_url}\" target=\"_blank\" rel=\"noopener noreferrer\">{newest_url}</a></li>"
                "</ul>"
            )

            if d.get("missing_date_results"):
                missing_items = []
                for r in d["missing_date_results"]:
                    title = html_lib.escape(str(r["title"]))
                    url = html_lib.escape(str(r["url"]))
                    missing_items.append(
                        f"<li>{title}<br>"
                        f"<span style='margin-left:0.5rem;'>URL: <a href=\"{url}\" target=\"_blank\" rel=\"noopener noreferrer\">{url}</a></span></li>"
                    )

                parts.append(
                    "<div style='margin-top:0.75rem;'><b>Results excluded due to missing <code>published_date</code></b></div>"
                    f"<ul style='margin-top:0.25rem; padding-left:1.2rem;'>{''.join(missing_items)}</ul>"
                )

        body_html = "\n".join(parts)

        display(HTML(f"""
                        <details style="margin: 0.5rem 0 0.75rem 0;">
                        <summary style="cursor:pointer; font-weight:600; font-size:1.00rem; line-height:1.2;">
                            {summary_text}
                        </summary>
                        <div style="margin-top:0.5rem;">
                            {body_html}
                        </div>
                        </details>
                    """))