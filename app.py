import streamlit as st
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from backend.graph import ask_medical_rag


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="ICU Clinical Dashboard",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 ICU Clinical Decision Dashboard")
st.caption("RAG-based evidence synthesis system (v2)")


# =========================
# SESSION STATE
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_debug" not in st.session_state:
    st.session_state.show_debug = False


show_debug = st.toggle("Debug mode", value=st.session_state.show_debug)
st.session_state.show_debug = show_debug


# =========================
# SEVERITY UI
# =========================

def get_severity_style(level: str):

    level = (level or "").lower()

    if level == "emergency":
        return {"color": "#ff4b4b", "label": "EMERGENCY", "icon": "🚨"}

    if level == "urgent":
        return {"color": "#ffb020", "label": "URGENT", "icon": "⚠️"}

    return {"color": "#2ecc71", "label": "ROUTINE", "icon": "🟢"}


# =========================
# RENDER FUNCTION
# =========================

def render_assistant_result(result: dict):

    answer = result.get("answer", "")
    triage = result.get("triage", {})

    severity = triage.get("urgency_level", "routine")
    style = get_severity_style(severity)

    scores = result.get("clinical_scores", {})

    # =========================
    # HEADER CARD
    # =========================

    st.markdown(
        f"""
        <div style="
            padding:15px;
            border-radius:10px;
            background-color:{style['color']}22;
            border-left:6px solid {style['color']};
            margin-bottom:15px;
        ">
            <h3 style="margin:0;">
                {style['icon']} {style['label']} CASE
            </h3>

            <p style="margin:0;">
                Clinical Domain:
                {triage.get('clinical_domain','Unknown')}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 2])

    # =========================
    # LEFT PANEL
    # =========================

    with col1:

        st.subheader("🧠 Triage Summary")

        st.metric(
            "Severity",
            severity.upper()
        )

        # ---------------------
        # FLAGS
        # ---------------------

        st.write("**Critical Flags**")

        flags = triage.get(
            "critical_flags",
            []
        )

        if flags:
            for f in flags:
                st.markdown(f"- `{f}`")
        else:
            st.write("None")

        # ---------------------
        # DOMAIN
        # ---------------------

        st.write("**Clinical Domain**")

        st.info(
            triage.get(
                "clinical_domain",
                "Unknown"
            )
        )

        # ---------------------
        # VALIDATION
        # ---------------------

        st.write("**Validation Status**")

        status = result.get(
            "validation_status",
            "unknown"
        )

        if status == "pass":
            st.success("PASS")
        elif status == "fail":
            st.error("FAIL")
        else:
            st.warning("UNKNOWN")

        reason = result.get(
            "validation_reason",
            ""
        )

        if reason:
            st.caption(reason)

        # ---------------------
        # EVIDENCE METRICS
        # ---------------------

        st.write("**Evidence Score**")

        st.metric(
            "Source Quality",
            round(
                result.get(
                    "source_quality_score",
                    0
                ),
                3
            )
        )

        st.metric(
            "Trusted Sources",
            result.get(
                "trusted_sources_count",
                0
            )
        )

        st.metric(
            "Domain Diversity",
            result.get(
                "domain_diversity",
                0
            )
        )

        # ---------------------
        # CLINICAL QUALITY MATRIX
        # ---------------------

        st.write("**Clinical Quality Matrix**")

        if scores:

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "Acute Stabilization",
                    f"{scores.get('acute_stabilization',0)}/10"
                )

                st.metric(
                    "Investigation Selection",
                    f"{scores.get('investigation_selection',0)}/10"
                )

                st.metric(
                    "Hypotension Safety",
                    f"{scores.get('safety_for_hypotensive_patient',0)}/10"
                )

            with c2:

                st.metric(
                    "Condition Management",
                    f"{scores.get('condition_management',0)}/10"
                )

                st.metric(
                    "Guideline Consistency",
                    f"{scores.get('guideline_consistency',0)}/10"
                )

                st.metric(
                    "Overall",
                    f"{scores.get('overall',0)}/10"
                )

        else:
            st.info(
                "No clinical quality scores available."
            )

        if show_debug:

            st.write("**Search Query**")

            st.code(
                result.get(
                    "search_query",
                    ""
                )
            )

    # =========================
    # RIGHT PANEL
    # =========================

    with col2:

        st.subheader(
            "📋 Clinical Recommendation"
        )

        st.markdown(answer)

    # =========================
    # DEBUG PANEL
    # =========================

    if show_debug:

        with st.expander(
            "🧪 Debug Panel",
            expanded=False
        ):

            st.subheader(
                "Full Triage JSON"
            )

            st.json(triage)

            st.subheader(
                "Clinical Scores"
            )

            st.json(scores)

            st.subheader(
                "Validation Reason"
            )

            st.write(
                result.get(
                    "validation_reason",
                    ""
                )
            )

            st.subheader(
                "Retrieved Sources"
            )

            docs = result.get(
                "retrieved_docs",
                []
            )

            for i, d in enumerate(docs):

                st.markdown(
                    f"### Source {i+1}"
                )

                st.write(
                    d.get(
                        "url",
                        ""
                    )
                )

                st.write(
                    d.get(
                        "content",
                        ""
                    )[:500]
                )

# =========================
# CHAT HISTORY
# =========================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        if msg["role"] == "user":
            st.markdown(msg["content"])

        else:
            render_assistant_result(msg["result"])


# =========================
# INPUT
# =========================

prompt = st.chat_input("Enter clinical query...")

if prompt:

    # store user
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # run backend
    with st.chat_message("assistant"):

        with st.spinner("Running ICU RAG pipeline..."):

            result = ask_medical_rag(prompt)

            render_assistant_result(result)

    # store assistant (FIXED STRUCTURE)
    st.session_state.messages.append({
        "role": "assistant",
        "result": result
    })