from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import urlparse
from collections import Counter
import json
import pandas as pd
from helpers import _extract_domain, _clean_llm_json
from evaluations.llm_as_a_judge import LLMJudge
from models import CoverageResult


def compute_breadth_score(num_facets: int, results_count: int, top_facet_share: float) -> float:
    """
    0-100 breadth score

    num_facets: number of facets with count > 0
    results_count: total results returned
    top_facet_share: max_facet_count / results_count
    """
    if results_count <= 0:
        return 0.0
    if num_facets <= 0:
        return 0.0

    cap = min(results_count, 6)
    richness = min(num_facets, cap) / cap  # 0..1

    if num_facets == 1:
        penalty = 1.0
    else:
        ideal_share = 1.0 / num_facets
        penalty = (top_facet_share - ideal_share) / (1.0 - ideal_share)
        penalty = max(0.0, min(1.0, penalty))

    evenness = 1.0 - penalty  # 0..1

    score = 100.0 * (0.7 * richness + 0.3 * evenness)
    return round(score, 1)

class CoverageBreadthJudge:
    """
    Uses an LLM to infer which 'facets/subtopics' are covered by a set of search results.

    Key method:
        evaluate_source_coverage(search_calls) -> (summary_df, details)

    Notes:
    - Runs 1 LLM call per tavily_search tool call.
    - Uses only title/snippet/domain/url;
    """

    def __init__(
        self,
        llm_judge: LLMJudge,
        max_facets: int = 8,
    ):
        self.llm_judge = llm_judge
        self.max_facets = max_facets
        base_dir = Path(__file__).parent
        prompts_dir = base_dir / "prompts" 
        self._prompt_template = (prompts_dir/"coverage_breadth_prompt.txt").read_text(encoding="utf-8")


    def _build_prompt(self, query: str, results: List[Dict[str, Any]]) -> str:
        # keep compact to reduce tokens
        packed = []
        for i, r in enumerate(results, start=1):
            packed.append({
                "rank": i,
                "title": r.get("title", ""),
                "domain": _extract_domain(r.get("url")),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
            })

        schema = {
            "facets": ["facet_1", "facet_2"],
            "result_facets": [
                {"rank": 1, "facets": ["facet_1"], "why": "<short description of why result is in facet>"},
            ],
        }
        return self._prompt_template.format(
            max_facets=self.max_facets,
            schema_json=json.dumps(schema, indent=2),
            query=query,
            results_json=json.dumps(packed, ensure_ascii=False),
        ).strip()

    def judge_call(self, query: str, results: List[Dict[str, Any]]) -> CoverageResult:
        prompt = self._build_prompt(query=query, results=results)
        raw = self.llm_judge.generate(prompt)
        cleaned = _clean_llm_json(raw)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Coverage judge returned invalid JSON.\nRaw:\n{raw}\n\nCleaned:\n{cleaned}") from e

        if "facets" not in parsed or "result_facets" not in parsed:
            raise ValueError(f"Coverage judge output missing required keys. Output:\n{cleaned}")

        facets = [str(f).strip() for f in parsed["facets"] if str(f).strip()]
        facets = facets[: self.max_facets]

        # Normalize assignments and clamp facet count to 1-3 each
        assignments = []
        for item in parsed["result_facets"]:
            rank = item.get("rank")
            fs = item.get("facets", [])
            fs = [str(f).strip() for f in fs if str(f).strip()]
            fs = fs[:3]
            why = str(item.get("why", "")).strip()
            assignments.append({"rank": rank, "facets": fs, "why": why})

        # Count facets based on assignments (not just the top list)
        facet_counts = Counter()
        for a in assignments:
            for f in a["facets"]:
                facet_counts[f] += 1

        # Ensure facet list includes anything used in assignments
        for f in facet_counts.keys():
            if f not in facets:
                facets.append(f)
        facets = facets[: self.max_facets]

        # Recompute counts restricted to final facets list (optional)
        facet_counts = {f: int(facet_counts.get(f, 0)) for f in facets}

        return CoverageResult(
            facets=facets,
            facet_assignments=assignments,
            facet_counts=facet_counts,
        )

    def evaluate_source_coverage(
        self,
        search_calls: List[Dict[str, Any]],
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Evaluate coverage breadth per tavily_search call.
        Returns summary df + detailed objects for reporting.
        """
        rows: List[Dict[str, Any]] = []
        details: List[Dict[str, Any]] = []

        for idx, call in enumerate(search_calls, start=1):
            call_id = call.get("tool_call_id")
            results = call.get("results", []) or []

            # Try to recover query from content JSON if present
            query = ""
            content_obj = call.get("content")
            if isinstance(content_obj, str):
                try:
                    parsed = json.loads(content_obj)
                    query = parsed.get("query", "") or ""
                except Exception:
                    pass

            if not results:
                row = {
                    "tool_call_index": idx,
                    "tool_call_id": call_id,
                    "results_count": 0,
                    "num_facets": 0,
                    "breadth_score": None,
                    "top_facet": None,
                    "top_facet_share": None,
                    "note": "No results returned.",
                    "query": query,
                }
                rows.append(row)
                details.append({**row, "facets": [], "facet_counts": {}, "per_result": []})
                continue

            try:
                judged = self.judge_call(query=query, results=results)
            except Exception as e:
                row = {
                    "tool_call_index": idx,
                    "tool_call_id": call_id,
                    "results_count": len(results),
                    "num_facets": None,
                    "breadth_score": None,
                    "top_facet": None,
                    "top_facet_share": None,
                    "note": f"Coverage judging failed: {e}",
                    "query": query,
                }
                rows.append(row)
                details.append({**row, "facets": [], "facet_counts": {}, "per_result": [], "error": str(e)})
                continue

            facet_counts = judged.facet_counts
            num_facets = len([f for f, c in facet_counts.items() if c > 0])

            counts = [c for c in facet_counts.values() if c > 0]
            
            top_facet = None
            top_share = None
            if counts:
                top_facet, top_count = max(facet_counts.items(), key=lambda kv: kv[1])
                top_share = round(top_count / max(1, len(results)), 3)

            breadth_score = compute_breadth_score(
                num_facets=num_facets,
                results_count=len(results),
                top_facet_share=top_share
            )

            row = {
                "tool_call_index": idx,
                "tool_call_id": call_id,
                "results_count": len(results),
                "num_facets": num_facets,
                "breadth_score": breadth_score,
                "top_facet": top_facet,
                "top_facet_share": top_share,
                "note": None,
                "query": query,
            }
            rows.append(row)

            # Join assignments back to results for reporting
            per_result = []
            by_rank = {a["rank"]: a for a in judged.facet_assignments}
            for r_i, r in enumerate(results, start=1):
                a = by_rank.get(r_i, {"facets": [], "why": ""})
                per_result.append({
                    "rank": r_i,
                    "title": r.get("title"),
                    "domain": _extract_domain(r.get("url")),
                    "url": r.get("url"),
                    "facets": a.get("facets", []),
                    "why": a.get("why", ""),
                })

            details.append({
                **row,
                "facets": judged.facets,
                "facet_counts": facet_counts,
                "per_result": per_result,
            })

        summary_df = pd.DataFrame(rows)
        preferred_cols = [
            "tool_call_index",
            "tool_call_id",
            "results_count",
            "num_facets",
            "breadth_score",
            "top_facet",
            "top_facet_share",
            "note",
            "query",
        ]
        summary_df = summary_df[[c for c in preferred_cols if c in summary_df.columns]]
        return summary_df, details

def render_coverage_breadth_report(summary_df: pd.DataFrame, details: List[Dict[str, Any]]) -> None:
    from IPython.display import display, HTML
    import html as html_lib

    display(summary_df)

    for d in details:
        summary_text = html_lib.escape(f"Search #{d['tool_call_index']} — {d['tool_call_id']}")

        parts = []

        if d.get("query"):
            parts.append(f"<div><b>Query:</b> {html_lib.escape(str(d['query']))}</div>")

        if d.get("note"):
            parts.append(f"<div style='margin-top:0.35rem;'><b>Note:</b> {html_lib.escape(str(d['note']))}</div>")

            body_html = "\n".join(parts)
            display(HTML(f"""
                            <details style="margin: 0.5rem 0 0.75rem 0;">
                            <summary style="cursor:pointer; font-weight:600; font-size:1.05rem; line-height:1.4;">
                                {summary_text}
                            </summary>
                            <div style="margin-top:0.5rem;">
                                {body_html}
                            </div>
                            </details>
                        """))
            continue

        parts.append(
            "<div style='margin-top:0.5rem;'>"
            f"<b>Breadth score:</b> {d.get('breadth_score')}/100<br>"
            f"<b>Facets detected:</b> {d.get('num_facets')}<br>"
            f"<b>Top facet share:</b> {d.get('top_facet_share')} ({html_lib.escape(str(d.get('top_facet')))})"
            "</div>"
        )

        facet_counts = d.get("facet_counts") or {}
        if facet_counts:
            sorted_facets = sorted(facet_counts.items(), key=lambda x: (-x[1], x[0]))
            facet_items = "\n".join(
                f"<li><code>{html_lib.escape(str(facet))}</code>: {count}</li>"
                for facet, count in sorted_facets
            )
            parts.append(
                "<div style='margin-top:0.75rem;'><b>Facet distribution</b></div>"
                f"<ul style='margin-top:0.35rem; padding-left:1.2rem;'>{facet_items}</ul>"
            )

        per_result = d.get("per_result") or []
        if per_result:
            per_items = []
            for r in per_result:
                rank = r.get("rank")
                title = html_lib.escape(str(r.get("title", "")))
                domain = html_lib.escape(str(r.get("domain", "")))
                url = html_lib.escape(str(r.get("url", "")))
                why = html_lib.escape(str(r.get("why", "")))
                facets = r.get("facets") or []

                if facets:
                    facets_html = ", ".join(f"<code>{html_lib.escape(str(f))}</code>" for f in facets)
                else:
                    facets_html = "None"

                per_items.append(f"""
                                    <li style="margin-bottom:0.75rem;">
                                    <div><b>[{rank}] {title}</b> <code>({domain})</code></div>
                                    <div style="margin-top:0.15rem;">
                                        <b>Facets:</b> {facets_html}<br>
                                        <b>Why:</b> {why}<br>
                                        <b>URL:</b> <a href="{url}" target="_blank" rel="noopener noreferrer">{url}</a>
                                    </div>
                                    </li>
                                """)

            parts.append("<div style='margin-top:0.75rem;'><b>Per-result facet assignments</b></div>")
            parts.append("<ul style='margin-top:0.35rem; padding-left:1.2rem;'>" + "\n".join(per_items) + "</ul>")

        body_html = "\n".join(parts)

        display(HTML(f"""
                        <details style="margin: 0.5rem 0 0.75rem 0;">
                        <summary style="cursor:pointer; font-weight:600; font-size:1.0rem; line-height:1.2;">
                            {summary_text}
                        </summary>
                        <div style="margin-top:0.5rem;">
                            {body_html}
                        </div>
                        </details>
                    """))