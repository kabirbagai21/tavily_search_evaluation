import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
from helpers import _clean_llm_json, _extract_domain
from evaluations.llm_as_a_judge import LLMJudge


class SourceQualityJudge:
    """
    LLM-as-a-judge for evaluating Tavily search domain quality.
    """

    def __init__(
        self,
        llm_judge: LLMJudge,
        reputability_weight: float = 0.7,
        relevance_weight: float = 0.3,
    ):
        self.llm_judge = llm_judge
        self.reputability_weight = reputability_weight
        self.relevance_weight = relevance_weight
        base_dir = Path(__file__).parent
        prompts_dir = base_dir / "prompts" 
        self._prompt_template = (prompts_dir/"source_quality_prompt.txt").read_text(encoding="utf-8")


    def _build_prompt(self, query: str, result: Dict[str, Any]) -> str:
        domain = _extract_domain(result.get("url"))
        title = result.get("title", "")
        url = result.get("url", "")
        snippet = result.get("content", "")
        tavily_score = result.get("score")
        
        return self._prompt_template.format(
            reputability_weight=self.reputability_weight,
            relevance_weight=self.relevance_weight,
            query=query,
            domain=domain,
            url=url,
            title=title,
            snippet=snippet,
            tavily_score=tavily_score,
        ).strip()

    def judge_result(self, query: str, result: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_prompt(query=query, result=result)
        raw = self.llm_judge.generate(prompt)
        cleaned = _clean_llm_json(raw)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM judge did not return valid JSON. Raw output:\n{raw}") from e

        required = [
            "reputability_score",
            "relevance_score",
            "overall_quality_score",
            "source_type",
            "is_ugc",
            "rationale",
        ]
        missing = [k for k in required if k not in parsed]
        if missing:
            raise ValueError(f"LLM judge output missing fields: {missing}. Raw output:\n{raw}")

        return parsed
        
    def evaluate_source_quality(
        self,
        search_calls: List[Dict[str, Any]],
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Evaluate source quality for Tavily search calls using an LLM judge.

        Parameters
        ----------
        search_calls : list[dict]
            Usually:
            [tr for tr in tool_results if tr.get("name") == "tavily_search"]

        Returns
        -------
        summary_df : pd.DataFrame
            One row per Tavily call.
        details : list[dict]
            Rich per-call details including per-result judgments.
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
                    parsed = json.loads(content_obj)
                    query = parsed.get("query")
                except Exception:
                    query = None
            if query is None:
                query = ""

            if not results:
                row = {
                    "tool_call_index": idx,
                    "tool_call_id": call_id,
                    "results_count": 0,
                    "avg_quality_score": None,
                    "avg_reputability_score": None,
                    "avg_relevance_score": None,
                    "high_quality_sources": 0,
                    "ugc_sources": 0,
                    "lowest_quality_source": None,
                    "highest_quality_source": None,
                    "note": "No search results returned.",
                    "query": query,
                }
                rows.append(row)
                details.append({**row, "judged_results": []})
                continue

            judged_results = []
            failures = []

            for r in results:
                try:
                    judgment = self.judge_result(query=query, result=r)
                    judged_results.append({
                        "title": r.get("title"),
                        "url": r.get("url"),
                        "domain": _extract_domain(r.get("url")),
                        "snippet": r.get("content"),
                        "tavily_score": r.get("score"),
                        **judgment,
                    })
                except Exception as e:
                    failures.append({
                        "title": r.get("title"),
                        "url": r.get("url"),
                        "error": str(e),
                    })

            if not judged_results:
                row = {
                    "tool_call_index": idx,
                    "tool_call_id": call_id,
                    "results_count": len(results),
                    "avg_quality_score": None,
                    "avg_reputability_score": None,
                    "avg_relevance_score": None,
                    "high_quality_sources": 0,
                    "ugc_sources": 0,
                    "lowest_quality_source": None,
                    "highest_quality_source": None,
                    "note": "All judgments failed.",
                    "query": query,
                }
                rows.append(row)
                details.append({
                    **row,
                    "judged_results": [],
                    "failures": failures,
                })
                continue

            avg_quality = round(
                sum(j["overall_quality_score"] for j in judged_results) / len(judged_results), 1
            )
            avg_reputability = round(
                sum(j["reputability_score"] for j in judged_results) / len(judged_results), 1
            )
            avg_relevance = round(
                sum(j["relevance_score"] for j in judged_results) / len(judged_results), 1
            )

            high_quality_sources = sum(1 for j in judged_results if j["overall_quality_score"] >= 4)
            ugc_sources = sum(1 for j in judged_results if bool(j["is_ugc"]))

            highest = max(judged_results, key=lambda x: x["overall_quality_score"])
            lowest = min(judged_results, key=lambda x: x["overall_quality_score"])

            note_parts = []
            if failures:
                note_parts.append(f"{len(failures)} judgment(s) failed")
            note = "; ".join(note_parts) if note_parts else None

            row = {
                "tool_call_index": idx,
                "tool_call_id": call_id,
                "results_count": len(judged_results),
                "avg_quality_score": avg_quality,
                "avg_reputability_score": avg_reputability,
                "avg_relevance_score": avg_relevance,
                "high_quality_sources": high_quality_sources,
                "ugc_sources": ugc_sources,
                "lowest_quality_source": lowest["domain"] or lowest["title"],
                "highest_quality_source": highest["domain"] or highest["title"],
                "note": note,
                "query": query,
            }

            rows.append(row)
            details.append({
                **row,
                "judged_results": judged_results,
                "failures": failures,
            })

        summary_df = pd.DataFrame(rows)

        preferred_cols = [
            "tool_call_index",
            "tool_call_id",
            "results_count",
            "avg_quality_score",
            "avg_reputability_score",
            "avg_relevance_score",
            "high_quality_sources",
            "ugc_sources",
            "highest_quality_source",
            "lowest_quality_source",
            "note",
            "query",
        ]
        summary_df = summary_df[[c for c in preferred_cols if c in summary_df.columns]]

        return summary_df, details

def render_source_quality_report(summary_df: pd.DataFrame, details: List[Dict[str, Any]]) -> None:
    from IPython.display import display, HTML
    import html as html_lib

    display(summary_df)

    for d in details:
        summary_text = html_lib.escape(f"Search #{d['tool_call_index']} — {d['tool_call_id']}")

        parts = []

        # Query + note
        if d.get("query"):
            parts.append(f"<div><b>Query:</b> {html_lib.escape(str(d['query']))}</div>")

        if d.get("note"):
            parts.append(f"<div style='margin-top:0.35rem;'><b>Note:</b> {html_lib.escape(str(d['note']))}</div>")

        # If no judgments, render what we have and move on
        if not d.get("judged_results"):
            body_html = "\n".join(parts) if parts else "<div>(No judged results.)</div>"
            display(HTML(f"""
                    <details style="margin: 0.5rem 0 0.75rem 0;">
                    <summary style="cursor:pointer; font-weight:600; font-size:1.05rem; line-height:1.4;">
                        {summary_text}
                    </summary>
                    <div style="margin-top:0.5rem;">{body_html}</div>
                    </details>
                    """))
            continue

        # Summary stats
        parts.append(
            "<div style='margin-top:0.5rem;'>"
            f"<b>Average quality:</b> {d.get('avg_quality_score')}/5<br>"
            f"<b>Average reputability:</b> {d.get('avg_reputability_score')}/5<br>"
            f"<b>Average relevance:</b> {d.get('avg_relevance_score')}/5<br>"
            f"<b>High-quality sources (>=4):</b> {d.get('high_quality_sources')}/{d.get('results_count')}<br>"
            f"<b>UGC sources:</b> {d.get('ugc_sources')}/{d.get('results_count')}"
            "</div>"
        )

        # Per-source judgments (sorted by overall quality desc)
        sorted_results = sorted(
            d["judged_results"],
            key=lambda x: x.get("overall_quality_score", -1),
            reverse=True,
        )

        # Build HTML list
        items = []
        for r in sorted_results:
            title = html_lib.escape(str(r.get("title", "")))
            domain = html_lib.escape(str(r.get("domain", "")))
            url = html_lib.escape(str(r.get("url", "")))
            stype = html_lib.escape(str(r.get("source_type", "")))
            rationale = html_lib.escape(str(r.get("rationale", "")))

            oq = r.get("overall_quality_score")
            rep = r.get("reputability_score")
            rel = r.get("relevance_score")
            ugc = r.get("is_ugc")

            items.append(f"""
                            <li style="margin-bottom:0.75rem;">
                            <div><b>{title}</b></div>
                            <div style="margin-top:0.15rem;">
                                <b>Domain:</b> <code>{domain}</code><br>
                                <b>Type:</b> <code>{stype}</code><br>
                                <b>Quality:</b> {oq}/5<br>
                                <b>Reputability:</b> {rep}/5<br>
                                <b>Relevance:</b> {rel}/5<br>
                                <b>UGC:</b> {html_lib.escape(str(ugc))}<br>
                                <b>URL:</b> <a href="{url}" target="_blank" rel="noopener noreferrer">{url}</a><br>
                                <b>Why:</b> {rationale}
                            </div>
                            </li>
                        """)

        parts.append("<div style='margin-top:0.75rem;'><b>Per-source judgments</b></div>")
        parts.append("<ul style='margin-top:0.35rem; padding-left:1.2rem;'>" + "\n".join(items) + "</ul>")

        # Failures (if any)
        if d.get("failures"):
            fail_items = []
            for f in d["failures"]:
                f_title = html_lib.escape(str(f.get("title", "")))
                f_url = html_lib.escape(str(f.get("url", "")))
                f_err = html_lib.escape(str(f.get("error", "")))
                fail_items.append(f"""
                                    <li>
                                    <div><b>{f_title}</b></div>
                                    <div style="margin-top:0.15rem;">
                                        <b>URL:</b> <a href="{f_url}" target="_blank" rel="noopener noreferrer">{f_url}</a><br>
                                        <b>Error:</b> <code>{f_err}</code>
                                    </div>
                                    </li>
                                """)
            parts.append("<div style='margin-top:0.75rem;'><b>Failed judgments</b></div>")
            parts.append("<ul style='margin-top:0.35rem; padding-left:1.2rem;'>" + "\n".join(fail_items) + "</ul>")

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