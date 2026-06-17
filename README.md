# Medical_Agent_Plasmit

# ICU Clinical Decision Dashboard

An evidence-grounded Retrieval-Augmented Generation (RAG) system for clinical question answering, triage assessment, source validation, and structured ICU-style recommendation generation.

The system combines:

* LangGraph workflow orchestration
* OpenAI GPT models
* Tavily web retrieval
* Multi-stage evidence scoring
* Clinical quality evaluation
* Streamlit dashboard interface

---

# Overview

This project is designed to simulate a clinical evidence synthesis pipeline rather than a traditional chatbot.

Given a clinical query, the system:

1. Performs clinical triage
2. Rewrites the query for medical retrieval
3. Retrieves evidence from medical sources
4. Scores source quality and trustworthiness
5. Generates an ICU-style recommendation
6. Evaluates clinical completeness
7. Validates output quality
8. Retries automatically when quality thresholds are not met

---

# Architecture

```text
User Query
    │
    ▼
Clinical Triage
    │
    ▼
Query Rewriting
    │
    ▼
Search Query Expansion
    │
    ▼
Tavily Medical Retrieval
    │
    ▼
Evidence Scoring
    │
    ▼
RAG Clinical Synthesis
    │
    ▼
Clinical Quality Judge
    │
    ▼
Validation Gate
    │
 ┌──┴─────┐
 │        │
Pass    Fail
 │        │
 ▼        ▼
Output   Retry/Fallback
```

---

# Features

## Clinical Triage

Automatically classifies:

* Emergency
* Urgent
* Routine

Extracts:

* Clinical domain
* Critical safety flags

Example:

```json
{
  "urgency_level": "emergency",
  "clinical_domain": "gastroenterology",
  "critical_flags": [
    "hypotension",
    "severe_anemia",
    "active_bleeding"
  ]
}
```

---

## Query Rewriting

Transforms natural language into a high-recall medical search query optimized for:

* PubMed
* WHO
* CDC
* NHS

Example:

Input:

```text
Patient has acute GI bleeding with CKD stage 4 and severe anemia.
```

Output:

```text
acute gastrointestinal bleeding CKD stage 4 severe anemia emergency management stabilization guidelines
```

---

## Evidence Retrieval

Uses Tavily Search to retrieve relevant medical information.

Current retrieval settings:

```python
TavilySearch(max_results=12)
```

Each document contains:

```python
{
    "url": "...",
    "content": "..."
}
```

---

## Source Trust Scoring

Trusted domains are weighted more heavily.

Examples:

* who.int
* cdc.gov
* mayoclinic.org
* pubmed.ncbi.nlm.nih.gov
* nejm.org
* jamanetwork.com
* bmj.com

Source quality combines:

```text
Domain Trust
+
Content Quality
+
Domain Diversity
-
Variance Penalty
-
Retrieval Stability Penalty
```

Output metrics:

```text
Source Quality Score
Trusted Sources Count
Domain Diversity
```

---

## ICU Clinical Recommendation Generation

Generates structured recommendations using only retrieved evidence.

Output sections:

```text
acute_stabilization
condition_management
investigation_selection
guideline_consistency
safety_for_hypotensive_patient
overall
references
```

The model is explicitly restricted to:

* Retrieved sources only
* No external knowledge
* Citation-based recommendations

---

## Clinical Quality Evaluation

A second LLM evaluates answer quality.

Scoring categories:

| Metric                         | Description                      |
| ------------------------------ | -------------------------------- |
| Acute Stabilization            | ABC management and resuscitation |
| Condition Management           | Diagnosis and severity           |
| Investigation Selection        | Labs, imaging, procedures        |
| Guideline Consistency          | Workflow correctness             |
| Safety for Hypotensive Patient | Shock recognition and escalation |

Example:

```json
{
  "acute_stabilization": 9,
  "condition_management": 8,
  "investigation_selection": 8,
  "guideline_consistency": 8,
  "safety_for_hypotensive_patient": 9,
  "overall": 8.4
}
```

---

## Validation Layer

Outputs are automatically validated.

Validation checks:

### Evidence Quality

```python
source_quality_score >= 0.20
```

### Retrieval Quality

```python
retrieved_docs >= 3
```

### Clinical Quality

```python
overall >= 6
```

If validation fails:

```text
Retry → Regenerate → Revalidate
```

Maximum retries:

```python
MAX_RETRIES = 2
```

Fallback response:

```text
Insufficient clinical evidence after multiple retrieval attempts.
```

---

# Streamlit Dashboard

The frontend provides:

## Triage Summary

* Severity Level
* Critical Flags
* Clinical Domain

## Evidence Metrics

* Source Quality Score
* Trusted Sources
* Domain Diversity

## Clinical Quality Matrix

* Acute Stabilization
* Condition Management
* Investigation Selection
* Guideline Consistency
* Hypotension Safety
* Overall Score

## Clinical Recommendation Panel

Displays the final structured recommendation.

## Debug Mode

Displays:

* Search query
* Triage JSON
* Clinical scores
* Validation reasoning
* Retrieved sources

---

# Technology Stack

## Backend

* LangGraph
* LangChain
* OpenAI GPT-5 Nano
* Tavily Search
* Pydantic

## Frontend

* Streamlit

## Monitoring

* LangSmith

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/icu-clinical-dashboard.git
cd icu-clinical-dashboard
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
```

---

# Running the Application

Start Streamlit:

```bash
streamlit run ui/app.py
```

Open:

```text
http://localhost:8501
```

---

# Example Query

```text
Patient is having acute GI bleeding, CKD stage 4, hypotension, and severe anemia with hemoglobin 5.8 g/dL. What investigations and management are recommended?
```

---

# Current Limitations

* Depends on retrieved web evidence quality
* Not a substitute for clinical judgement
* Search quality affects downstream performance
* Limited to publicly available sources

---

# Future Improvements

* PubMed API integration
* Medical knowledge graph retrieval
* Guideline-specific retrieval (NICE, KDIGO, ACG, SCCM)
* Citation verification layer
* Multi-agent clinical review workflow
* Human-in-the-loop approval
* Clinical audit logging

---

# Disclaimer

This project is intended for research and educational purposes only.

It is not a medical device and must not be used for real-world diagnosis, treatment decisions, or patient management without qualified clinical supervision.
