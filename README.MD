# Tavily Search Evaluation Suite

## Overview

This project provides a lightweight evaluation framework for analyzing the quality of search results returned by the Tavily Search API in LLM-powered workflows. Modern AI agents frequently rely on search APIs to retrieve information for reasoning, summarization, or question answering. However, the usefulness of these agents depends heavily on the quality, diversity, freshness, and coverage of the retrieved sources.

This repository introduces a set of evaluation tools designed to analyze Tavily search outputs using a combination of **metadata analysis and LLM-based judging**. The framework works directly with **LangGraph traces**, making it easy to evaluate search behavior in real-world agent pipelines.

The evaluations produce both **structured metrics (DataFrames)** and **interactive notebook reports**, enabling quick inspection and debugging of search quality.


## Business Value / Use Cases

These evaluations help organizations understand and improve the reliability of AI systems that rely on web search.

- **AI Agent Reliability**: AI assistants, copilots, and autonomous agents often use search APIs to gather external knowledge. Poor retrieval quality can lead to hallucinations, outdated information, or biased answers. These evaluations help identify weaknesses in the retrieval layer.

- **Retrieval Quality Monitoring**: Teams deploying production AI systems can use these metrics to continuously monitor search quality across queries, workflows, or time periods.

- **Search Provider Benchmarking**: Organizations comparing multiple search providers (e.g., Tavily vs other APIs) can use this framework to measure differences in freshness, diversity, and coverage.


# Evaluation Metrics

The evaluation suite currently includes four complementary retrieval metrics.



## 1. Freshness Evaluation

Measures how recent the retrieved sources are relative to a specified time window.

### Why it matters
Many queries require up-to-date information (news, technology developments, regulations, etc.). Stale results can lead to outdated or incorrect answers.

### Metrics produced

- **Freshness score (0–100)** based on result age and time window
- **Oldest and newest source**
- **Number of sources within the requested timeframe**
- **Missing publication date detection**

### Implementation Details

1. Extract `published_date` from Tavily search results.
2. Convert publication timestamps to age relative to the evaluation time.
3. Compute a freshness score using:
   - number of results within the timeframe
   - average recency of the results.

If publication dates are partially missing, the score is computed using only dated results while reporting exclusions.


## 2. Source Diversity Evaluation

Measures how many distinct **domains** appear in the search results.

## Why it matters
Search results dominated by a single source can create informational bias and reduce the breadth of perspectives.

### Metrics produced

- **Total sources**
- **Unique source domains**
- **Unique source percentage**
- **Most frequent source (if repeated)**
- **Domain breakdown**

### Implementation Details

1. Extract domains from result URLs.
2. Count unique domains and frequency distribution.
3. Compute the percentage of unique sources relative to the total results returned.

This provides a quick signal of whether search results are **diverse or redundant**.

## 3. Source Quality Evaluation (LLM-as-a-Judge)

Uses an LLM to evaluate the credibility and relevance of each source.

### Why it matters
Not all sources are equally reliable. High-quality retrieval should prioritize **trusted journalism, academic publications, official documentation, and authoritative references** over low-quality or user-generated content.

### Metrics produced

- **Reputability score**
- **Relevance score**
- **Overall quality score**
- **Source type classification**
- **UGC (user-generated content) detection**
- **Average quality across results**

### Implementation Details

For each search result, the LLM judge evaluates:

1. **Reputability**
   - editorial standards
   - institutional credibility
   - publication authority

2. **Relevance**
   - alignment with the search query

3. **Overall quality score**
   - weighted combination of reputability and relevance.

The LLM also classifies the source type (e.g., academic, news, blog, social media etc).

## 4. Coverage Breadth Evaluation

Measures how many **distinct subtopics or facets** of a query are represented in the search results.

## Why it matters
Good search results should cover multiple perspectives or aspects of a topic rather than repeating the same information.

Example query:

> "Latest developments in AI"

Strong coverage might include:

- new model releases
- regulation
- hardware/chips
- enterprise adoption
- safety concerns

Weak coverage might only include articles about one specific event.

## Metrics produced

- **Number of facets detected**
- **Facet distribution across results**
- **Top facet dominance**
- **Coverage breadth score**

## Technical approach

An LLM analyzes result titles and snippets to:

1. Identify the main **facets/subtopics** present in the result set.
2. Assign one or more facets to each result.
3. Count how many results cover each facet.

A breadth score is then computed using:

- number of facets discovered
- dominance of the most common facet.

This provides a simple measure of **how well the search results cover the conceptual space of the query**.

## Outputs

Each evaluation produces:

- **Structured metrics (pandas DataFrames)** for programmatic analysis
- **Interactive notebook reports** with collapsible sections for easy inspection

This design allows both **automated monitoring pipelines** and **manual exploratory analysis**.

## Future Extensions

Potential future evaluations include:

- query–result relevance scoring
- ranking quality evaluation
- duplicate / redundancy detection
- answerability scoring

# Setup

1. **Install dependencies:**
```bash
python3 -m venv venv or python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Set up environment variables:**
Create a `.env` file in the project root with:
```
TAVILY_API_KEY=your_tavily_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_PROJECT=web-agent
```

## Running the Application and Notebook

```bash
python app.py
```

The API will start on `http://localhost:8000`

You may run the cells of the notebook within your IDE or by starting the Jupyter server using the following command: 
```bash
jupyter notebook
```

Run the cells in the notebook ```agent_eval.ipynb``` to perform the evaluation.
