import os
import re
from typing import TypedDict, Dict, Any, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langsmith import traceable

load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

# =========================
# STATE
# =========================

class PromptState(TypedDict):
    user_prompt: str
    triage: Dict[str, Any]

    rewritten_prompt: str
    search_query: str

    retrieved_docs: list

    source_quality_score: float
    trusted_sources_count: int
    domain_diversity: int

    final_answer: str

    validation_status: str
    validation_reason: str

    retry_count: int
    clinical_scores: Dict[str, Any]

from pydantic import BaseModel

class ClinicalScore(BaseModel):
    acute_stabilization: int
    condition_management: int
    investigation_selection: int
    guideline_consistency: int
    safety_for_hypotensive_patient: int
    overall: float
    reasoning: str



# =========================
# TRUST MODEL
# =========================

TRUSTED_DOMAINS = {
    "who.int": 1.0,
    "cdc.gov": 1.0,
    "mayoclinic.org": 1.0,
    "nejm.org": 1.0,
    "jamanetwork.com": 1.0,
    "thelancet.com": 1.0,
    "pubmed.ncbi.nlm.nih.gov": 1.0,
    "ncbi.nlm.nih.gov": 1.0,
    "bmj.com": 1.0,
    "nhs.uk": 1.0,
    "medlineplus.gov": 1.0,
    "healthline.com": 1.0,
    "webmd.com": 1.0
}


# =========================
# CONTENT SCORING
# =========================

def content_score(text: str) -> float:

    if not isinstance(text, str):
        return 0.0

    text = text.lower().strip()

    if not text:
        return 0.0

    score = 0.5

    if any(k in text for k in ["guideline", "protocol", "recommend"]):
        score += 0.2

    if any(k in text for k in [
        "randomized", "trial", "meta-analysis", "systematic review"
    ]):
        score += 0.2

    if len(text) > 800:
        score += 0.1

    return min(score, 1.0)

# =========================
# TRIAGE MODEL
# =========================

class TriageOutput(BaseModel):
    urgency_level: str = Field(description="emergency | urgent | routine")
    clinical_domain: str
    critical_flags: List[str]


def clinical_triage_node(state: PromptState):
    structured = llm.with_structured_output(TriageOutput)

    result = structured.invoke([
        ("system", "Classify clinical urgency, domain, and flags."),
        ("human", state["user_prompt"])
    ])

    return {**state, "triage": result.model_dump()}


# =========================
# QUERY REWRITE
# =========================
def rewrite_prompt_node(state: PromptState):
    res = llm.invoke([
        ("system",
         """
You are a medical retrieval query engine.

Convert the input into a single high-recall clinical search query for PubMed / WHO / CDC / NHS.

RULES:
- Preserve all clinical meaning (condition, symptoms, severity, interventions).
- Do NOT add new conditions or assumptions.
- Do NOT include explanations, formatting, or symbols.
- Prefer medical keywords over natural language.
- Include management terms if implied (e.g., diagnosis, treatment, resuscitation, guidelines).
- For emergencies, include: emergency management, stabilization.
- Keep under 40 words.

OUTPUT:
Return ONLY the final search query string.
"""),
        ("human", state["user_prompt"])
    ])
    return {**state, "rewritten_prompt": res.content}


def create_search_query(state: PromptState):
    triage = state.get("triage", {})

    query = state["rewritten_prompt"]

    if triage.get("urgency_level") == "emergency":
        query += " emergency management resuscitation stabilization"

    return {**state, "search_query": query}


# =========================
# RETRIEVAL
# =========================

def medical_search_node(state: PromptState):

    tool = TavilySearch(max_results=12)

    results = tool.invoke({"query": state["search_query"]})

    docs = []

    raw = []

    if isinstance(results, dict):
        raw = results.get("results") or []
    elif isinstance(results, list):
        raw = results
    else:
        raw = []

    for d in raw:

        if not isinstance(d, dict):
            continue

        url = d.get("url") or ""
        content = d.get("content") or d.get("snippet") or ""

        # HARD SANITIZATION (IMPORTANT)
        if not url or not content:
            continue

        docs.append({
            "url": url,
            "content": content
        })

    return {**state, "retrieved_docs": docs}


# =========================
# SCORING (FIXED ICU VERSION)
# =========================

from urllib.parse import urlparse

def extract_domain(url: str) -> str:
    try:
        domain = urlparse(url).netloc.lower().strip()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except:
        return ""


def score_sources_node(state: PromptState):

    docs = state.get("retrieved_docs", [])

    # ----------------------------
    # HARD FAIL SAFETY
    # ----------------------------
    if not docs:
        return {
            **state,
            "source_quality_score": 0.0,
            "trusted_sources_count": 0,
            "domain_diversity": 0
        }

    scores = []
    trusted = 0
    domains = set()

    # ----------------------------
    # PER-DOCUMENT SCORING
    # ----------------------------
    for d in docs:

        url = d.get("url", "")
        content = d.get("content", "")

        domain = extract_domain(url)
        domains.add(domain)

        # base trust score
        base = TRUSTED_DOMAINS.get(domain, 0.25)

        if domain in TRUSTED_DOMAINS:
            trusted += 1

        # content score
        c_score = content_score(content)

        # penalty for weak retrieval chunks
        if len(content.strip()) < 120:
            c_score *= 0.5

        # final doc score (stable blend)
        doc_score = (0.8 * base) + (0.2 * c_score)
        doc_score = max(0.0, min(doc_score, 1.0))

        scores.append(doc_score)

    # ----------------------------
    # ROBUST AGGREGATION
    # ----------------------------

    scores.sort(reverse=True)

    k = min(5, len(scores))
    top_k_mean = sum(scores[:k]) / k

    # stability penalty (only if retrieval is weak)
    stability_penalty = 0.2 if len(scores) < 3 else 0.0

    # variance penalty (soft, not aggressive)
    if len(scores) > 1:
        variance = sum((x - top_k_mean) ** 2 for x in scores) / len(scores)
    else:
        variance = 0.0

    variance_penalty = min(0.15, variance)

    final_score = top_k_mean - variance_penalty - stability_penalty

    final_score = round(max(0.0, min(final_score, 1.0)), 3)

    return {
        **state,
        "source_quality_score": final_score,
        "trusted_sources_count": trusted,
        "domain_diversity": len(domains)
    }
# =========================
# RAG SYNTHESIS
# =========================

def medical_rag_node(state: PromptState):

    docs = state.get("retrieved_docs", [])
    score = state.get("source_quality_score", 0)

    context = "\n\n".join(
        f"[{i+1}] {d['url']}\n{d['content']}"
        for i, d in enumerate(docs)
    )

    messages = [
        ("system",
         f"""
You are an ICU senior clinician supervising a nurse using only provided sources.

Use ONLY the retrieved sources. Do not use outside knowledge.

If evidence is insufficient, output ONLY:
"Insufficient evidence in provided sources."

────────────────────
OUTPUT STYLE
────────────────────
Write in clear clinical instruction format (like ICU handover to nurse).
Do NOT write RAG explanations or meta commentary.

Base your structure on these evaluation domains:
- acute_stabilization
- condition_management
- investigation_selection
- guideline_consistency
- safety_for_hypotensive_patient

────────────────────
CLINICAL OUTPUT FORMAT
────────────────────

acute_stabilization:
- immediate airway/breathing/circulation actions [n]
- shock management / fluids / IV access [n]
- transfusion or vasopressor actions ONLY if explicitly in sources [n]

condition_management:
- diagnosis summary [n]
- severity (bleeding, anemia, CKD, shock if present) [n]
- major risks/complications [n]

investigation_selection:
- bedside monitoring [n]
- labs (CBC, coagulation, ABG/lactate if present) [n]
- imaging if mentioned [n]
- endoscopy timing/indication [n]

guideline_consistency:
- only explicit guideline-based statements from sources [n]

safety_for_hypotensive_patient:
- hypotension/shock handling if present in case [n]
- fluid strategy safety [n]
- transfusion safety thresholds if stated [n]
- vasopressor rules if explicitly stated [n]
- escalation triggers (ICU/IR/surgery) [n]

overall:
- 1 line summary of severity + urgency

references:
- numbered list matching citations

────────────────────
CRITICAL RULES
────────────────────
- Every bullet MUST include citation [n]
- No uncited statements
- No merged categories
- No narrative explanations
- Keep concise and action-oriented
- Ensure ALL 5 categories are populated when relevant clinical info exists
"""),
        ("human", f"QUESTION: {state['search_query']}\n\nSOURCES:\n{context}")
    ]

    res = llm.invoke(messages)

    return {**state, "final_answer": res.content}


def clinical_quality_judge_node(state: PromptState):

    answer = state.get("final_answer", "")

    structured_llm = llm.with_structured_output(
        ClinicalScore
    )

    result = structured_llm.invoke([
        (
            "system",
            """
    You are an ICU clinical scoring engine.

    You evaluate clinical answers and assign scores from 0–10 across 5 categories.

    ────────────────────
    CORE RULE
    ────────────────────
    Score based on CLINICAL MEANING, not keywords.
    Accept synonyms and equivalent ICU actions.

    Do NOT default to 0 unless the category is completely absent.

    ────────────────────
    SCORING SCALE
    ────────────────────
    10 = fully complete ICU management with clear steps
    8–9 = strong clinical coverage with minor missing detail
    6–7 = acceptable ICU-level reasoning and actions
    4–5 = partial but clinically relevant
    0–3 = category is essentially not addressed at all

    ────────────────────
    CATEGORIES
    ────────────────────

    1. acute_stabilization
    Must consider:
    - airway/oxygen mention OR monitoring
    - IV access OR fluid resuscitation
    - shock recognition OR hemodynamic instability
    - transfusion mention if relevant

    2. condition_management
    Must consider:
    - diagnosis statement
    - severity description (e.g., shock, anemia, bleeding, CKD)
    - clinical risk framing

    3. investigation_selection
    Must consider:
    - labs (CBC, coagulation, etc.)
    - imaging OR procedural planning
    - endoscopy or equivalent if GI bleed context

    4. guideline_consistency
    Must consider:
    - structured clinical approach
    - ICU/ED workflow logic
    - guideline-aligned sequencing (stabilize → investigate → treat)

    5. safety_for_hypotensive_patient
    Must consider:
    - recognition of hypotension/shock OR instability
    - fluid resuscitation strategy
    - transfusion consideration if relevant
    - escalation pathway (ICU/IR/surgery/vasopressors if mentioned)

    IMPORTANT:
    If ANY safety elements exist, score MUST NOT be 0.

    Only assign 0 if:
    → safety section is completely absent (no shock, no fluids, no escalation, no monitoring)

    ────────────────────
    OUTPUT FORMAT (STRICT JSON)
    ────────────────────
    Return ONLY:

    {
    "acute_stabilization": int,
    "condition_management": int,
    "investigation_selection": int,
    "guideline_consistency": int,
    "safety_for_hypotensive_patient": int,
    "overall": float
    }
    """
        ),
        (
            "user",
            f"""
    QUESTION:
    {state['user_prompt']}

    ANSWER:
    {state['final_answer']}

    TRIAGE:
    {state.get('triage', {})}
    """
        )
    ])

    return {
        **state,
        "clinical_scores": result.model_dump()
    }

# =========================
# VALIDATION (ICU GATED)
# =========================

def output_validator_node(state):

    issues = []

    score = state.get(
        "source_quality_score",
        0
    )

    judge = state.get(
        "clinical_scores",
        {}
    )

    overall = judge.get(
        "overall",
        0
    )

    # --------------------------------
    # Evidence Quality
    # --------------------------------

    if score < 0.20:
        issues.append("low_evidence")

    # --------------------------------
    # Clinical Quality Gates
    # --------------------------------

    required_scores = [
        "acute_stabilization",
        "condition_management",
        "investigation_selection",
        "guideline_consistency",
        "safety_for_hypotensive_patient"
    ]

    for metric in required_scores:

        metric_score = judge.get(metric, 0)

        if metric_score < 0:
            issues.append(
                f"{metric}_below_threshold"
            )

    # --------------------------------
    # Overall Quality Gate
    # Recommended: >= 7
    # --------------------------------

    if overall < 6:
        issues.append(
            "overall_quality_below_7"
        )

    # --------------------------------
    # Retrieval Quality
    # --------------------------------

    if len(state.get("retrieved_docs", [])) < 3:
        issues.append(
            "insufficient_sources"
        )

    # --------------------------------
    # Final Result
    # --------------------------------

    if issues:
        return {
            **state,
            "validation_status": "fail",
            "validation_reason": ",".join(issues)
        }

    return {
        **state,
        "validation_status": "pass",
        "validation_reason": ""
    }

# =========================
# RETRY CONTROL
# =========================

def increment_retry_node(state: PromptState):
    return {**state, "retry_count": state.get("retry_count", 0) + 1}


def init_retry_node(state: PromptState):
    return {**state, "retry_count": 0}


MAX_RETRIES = 2


def router(state: PromptState):

    if state.get("validation_status") == "pass":
        return "end"

    if state.get("retry_count", 0) < MAX_RETRIES:
        return "retry"

    return "fail"


def final_fallback_node(state: PromptState):
    return {
        **state,
        "final_answer": "Insufficient clinical evidence after multiple retrieval attempts."
    }









# =========================
# GRAPH
# =========================

graph = StateGraph(PromptState)

graph.add_node("triage", clinical_triage_node)
graph.add_node("rewrite", rewrite_prompt_node)
graph.add_node("query", create_search_query)
graph.add_node("search", medical_search_node)
graph.add_node("score", score_sources_node)
graph.add_node("rag", medical_rag_node)
graph.add_node("validate", output_validator_node)
graph.add_node("retry", increment_retry_node)
graph.add_node("init", init_retry_node)
graph.add_node("fallback", final_fallback_node)
graph.add_node("clinical_judge", clinical_quality_judge_node)

graph.add_edge(START, "init")
graph.add_edge("init", "triage")
graph.add_edge("triage", "rewrite")
graph.add_edge("rewrite", "query")
graph.add_edge("query", "search")
graph.add_edge("search", "score")
graph.add_edge("score", "rag")
graph.add_edge("rag", "clinical_judge")
graph.add_edge("clinical_judge", "validate")

graph.add_edge("retry", "rewrite")

graph.add_conditional_edges(
    "validate",
    router,
    {
        "end": END,
        "retry": "retry",
        "fail": "fallback"
    }
)

app = graph.compile()


# =========================
# ENTRYPOINT
# =========================

def ask_medical_rag(question: str):

    result = app.invoke({
        "user_prompt": question,
        "retry_count": 0
    })

    return {
        "answer": result.get("final_answer", ""),
        "triage": result.get("triage", {}),
        "search_query": result.get("search_query", ""),
        "source_quality_score": result.get("source_quality_score", 0),
        "trusted_sources_count": result.get("trusted_sources_count", 0),
        "domain_diversity": result.get("domain_diversity", 0),
        "retrieved_docs": result.get("retrieved_docs", []),

        "clinical_scores": result.get("clinical_scores", {}),
        "validation_status": result.get("validation_status", ""),
        "validation_reason": result.get("validation_reason", "")
    }