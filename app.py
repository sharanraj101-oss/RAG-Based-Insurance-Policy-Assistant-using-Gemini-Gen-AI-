import sys
import os
import json
import time
from pathlib import Path
import streamlit as st
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import InsuranceAssistantPipeline
from src.agent_tools import lookup_claim_status, calculate_insurance_premium
from src.guardrails import GuardrailsManager
from src.config import DATA_DIR, LLM_MODEL_NAME

# Page Configuration
st.set_page_config(
    page_title="Insurance Policy AI Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Dark/Glassmorphism Aesthetic
st.markdown("""
<style>
    /* Main Background & Font Styling */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Glassmorphism Header Card */
    .header-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Title & Subtitle */
    .main-title {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 6px;
    }

    /* Metric Cards */
    .metric-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-box:hover {
        transform: translateY(-2px);
        border-color: #6366f1;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Citation Badges & Boxes */
    .citation-card {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #6366f1;
        border-radius: 6px;
        padding: 12px 16px;
        margin-top: 8px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }

    /* Guardrails Badge */
    .badge-pass {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid #22c55e;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-warn {
        background-color: rgba(234, 179, 8, 0.2);
        color: #facc15;
        border: 1px solid #eab308;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-tool {
        background-color: rgba(168, 85, 247, 0.2);
        color: #c084fc;
        border: 1px solid #a855f7;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Pipeline Instance in Session State
@st.cache_resource(show_spinner="Initializing RAG Policy Engine & Knowledge Base...")
def load_rag_pipeline():
    return InsuranceAssistantPipeline()

pipeline = load_rag_pipeline()

# Header Banner
st.markdown("""
<div class="header-card">
    <h1 class="main-title">🛡️ Smart RAG Insurance Policy Assistant</h1>
    <div class="sub-title">Powered by Gemini LLM • Hybrid Vector/BM25 Search • Cross-Encoder Reranking • Agentic Tools</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("⚙️ System Control & Settings")
st.sidebar.subheader("Retrieval Architecture")

retrieval_mode = st.sidebar.selectbox(
    "Select Retrieval Strategy",
    options=["hybrid_rerank", "hybrid", "vector_only", "bm25_only"],
    format_func=lambda x: {
        "hybrid_rerank": "⚡ Hybrid RRF + Cross-Encoder Rerank (Recommended)",
        "hybrid": "🔀 Hybrid RRF Search (Dense + BM25)",
        "vector_only": "🎯 Dense Vector Search (TF-IDF Cosine)",
        "bm25_only": "🔍 Sparse Keyword Search (BM25Okapi)"
    }[x]
)

top_k_chunks = st.sidebar.slider("Top K Retrieved Chunks", min_value=1, max_value=6, value=3)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Active Agentic Tools")
st.sidebar.markdown("""
- 🔍 **Live Claim Status Lookup**: Recognizes Claim IDs (e.g. `CLM-1001`, `CLM-1002`)
- 🧮 **Premium Calculator**: Automatic premium calculation based on age & coverage
- 🛡️ **Guardrails Safety**: Real-time PII redaction & groundedness verification
""")

st.sidebar.markdown("---")
st.sidebar.caption(f"LLM Engine: `{LLM_MODEL_NAME}` | Data KB: `4 Policy Categories`")

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💬 Policy Chatbot",
    "🔍 Claim Status Lookup",
    "🧮 Premium Calculator",
    "🛡️ Guardrails & Safety",
    "📊 RAG Benchmarks",
    "📚 Knowledge Base Explorer"
])

# ---------------------------------------------------------
# TAB 1: POLICY CHATBOT
# ---------------------------------------------------------
with tab1:
    st.subheader("💬 Ask Any Question About Insurance Policies")
    st.caption("Ask about waiting periods, coverage limits, exclusions, claim steps, or IRDAI regulations.")
    
    # Quick Sample Queries
    st.write("**Quick Test Prompts:**")
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    sample_q = None
    if col_q1.button("⏳ Pre-existing Waiting Period"):
        sample_q = "What is the waiting period for pre-existing disease?"
    if col_q2.button("🏥 Cashless Hospitalization"):
        sample_q = "What is covered under cashless hospitalization?"
    if col_q3.button("🚗 Motor Zero Dep Rider"):
        sample_q = "How is zero depreciation coverage applied for motor insurance?"
    if col_q4.button("📜 IRDAI Claim Settlement"):
        sample_q = "What is the claim settlement timeline according to IRDAI guidelines?"

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your AI Insurance Policy Assistant. How can I help you today with your health, motor, or life insurance queries?"
            }
        ]

    # Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "metadata" in msg:
                meta = msg["metadata"]
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.caption(f"⏱️ Latency: **{meta.get('latency_seconds')}s**")
                col_m2.caption(f"🎯 Groundedness Score: **{meta.get('groundedness_score')}**")
                col_m3.caption(f"📚 Retrieved Chunks: **{meta.get('retrieved_count')}**")

    # Chat Input
    prompt_input = st.chat_input("Ask a policy question or enter a Claim ID (e.g. CLM-1001)...")
    user_query = sample_q or prompt_input

    if user_query:
        # User message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Assistant response generation
        with st.chat_message("assistant"):
            with st.spinner("Analyzing query, searching policy knowledge base, and applying guardrails..."):
                response_data = pipeline.run_query(user_query, retrieval_mode=retrieval_mode, top_k=top_k_chunks)
                
                # Render Guardrails Banner if PII was redacted
                if response_data.get("redactions"):
                    st.markdown(f"<span class='badge-warn'>🛡️ PII Guardrail Triggered: {', '.join(response_data['redactions'])}</span>", unsafe_allow_html=True)
                    st.caption(f"Sanitized Query: `{response_data['sanitized_query']}`")

                # Render Tool Output Badge if tool was executed
                if response_data.get("tool_used"):
                    st.markdown(f"<span class='badge-tool'>🔧 Agentic Tool Invoked: `{response_data['tool_used']}`</span>", unsafe_allow_html=True)
                
                # Answer content
                st.markdown(response_data["answer"])

                # Render Citations expander
                if response_data.get("citations"):
                    with st.expander("📚 View Retrieved Policy Clause Citations"):
                        for idx, cite in enumerate(response_data["citations"]):
                            st.markdown(f"""
                            <div class="citation-card">
                                <strong>[{cite['category']} | Clause: {cite['clause']}]</strong> - {cite['title']}<br/>
                                <em style="color: #cbd5e1;">"{cite['snippet']}"</em><br/>
                                <small style="color: #64748b;">Source File: {cite['source']}</small>
                            </div>
                            """, unsafe_allow_html=True)

                # Render Metrics Footer
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("⏱️ Response Latency", f"{response_data['latency_seconds']}s")
                c2.metric("🛡️ Groundedness", f"{response_data['groundedness_score']}")
                c3.metric("📖 Retrieved Contexts", response_data["retrieved_count"])
                c4.metric("🔍 Search Strategy", response_data["retrieval_mode"])

                # Save assistant message with metadata
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_data["answer"],
                    "metadata": response_data
                })

# ---------------------------------------------------------
# TAB 2: CLAIM STATUS LOOKUP TOOL
# ---------------------------------------------------------
with tab2:
    st.subheader("🔍 Live Claim Status Tracker")
    st.caption("Look up realtime insurance claims stored in the insurer database.")
    
    col_c1, col_c2 = st.columns([2, 1])
    with col_c1:
        claim_input = st.text_input("Enter Claim Reference ID", value="CLM-1001", placeholder="e.g. CLM-1001, CLM-1002, CLM-1003")
        lookup_btn = st.button("🔎 Search Claim Database", type="primary")

    with col_c2:
        st.info("💡 **Test Claim IDs Available:**\n- `CLM-1001` (Health - Approved)\n- `CLM-1002` (Motor - Under Inspection)\n- `CLM-1003` (Life - Docs Required)")

    if lookup_btn or claim_input:
        claim_result = lookup_claim_status(claim_input)
        if claim_result.get("found"):
            data = claim_result["data"]
            st.success(f"Claim Record Found for ID: **{data['claim_id']}**")
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Claimant Name", data["claimant_name"])
            mc2.metric("Policy Number", data["policy_number"])
            mc3.metric("Claim Status", data["status"])
            
            st.json(data)
        else:
            st.error(claim_result.get("message"))

# ---------------------------------------------------------
# TAB 3: PREMIUM CALCULATOR TOOL
# ---------------------------------------------------------
with tab3:
    st.subheader("🧮 Instant Insurance Premium Calculator")
    st.caption("Calculate estimated annual premiums dynamically based on policy plan and coverage options.")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        age = st.slider("Customer Age (Years)", min_value=18, max_value=80, value=35)
        sum_insured = st.select_slider(
            "Sum Insured (Coverage Amount)",
            options=[300000.0, 500000.0, 1000000.0, 2000000.0, 5000000.0],
            format_func=lambda x: f"INR {x:,.0f}"
        )
    with col_p2:
        plan_type = st.radio("Select Policy Category", options=["Health", "Motor"], horizontal=True)
        zero_dep = st.checkbox("Include Zero-Depreciation / Rider Add-on Coverage", value=True)

    if st.button("🧮 Calculate Premium Quote", type="primary"):
        quote = calculate_insurance_premium(age, sum_insured, plan_type.lower(), zero_dep)
        
        st.markdown("### 📊 Premium Breakdown Quote")
        qp1, qp2, qp3, qp4 = st.columns(4)
        qp1.metric("Base Premium", quote["base_premium"])
        qp2.metric("Rider Cost", quote["rider_cost"])
        qp3.metric("GST (18%)", quote["gst_18_percent"])
        qp4.metric("Total Annual Payable", quote["total_annual_premium"])

        st.json(quote)

# ---------------------------------------------------------
# TAB 4: GUARDRAILS & SAFETY INSPECTOR
# ---------------------------------------------------------
with tab4:
    st.subheader("🛡️ Safety, PII Redaction & Hallucination Prevention")
    st.caption("Test how input guardrails redact sensitive personal data and verify groundedness.")

    guard_mgr = GuardrailsManager()
    
    st.markdown("#### 1. Realtime PII Redaction Tester")
    test_pii_text = st.text_area(
        "Enter sample text containing phone numbers, emails, Aadhaar numbers, or cards:",
        value="My name is Rahul. Call me at 9876543210 or email info@mycompany.com. My Aadhaar is 1234 5678 9012."
    )
    
    if st.button("🛡️ Test PII Redaction"):
        clean_txt, red_list = guard_mgr.sanitize_input(test_pii_text)
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("**Original Input Text:**")
            st.code(test_pii_text)
        with col_g2:
            st.markdown("**Sanitized Text (Post-Redaction):**")
            st.code(clean_txt)
        st.markdown(f"**Redactions Applied:** `{red_list}`")

# ---------------------------------------------------------
# TAB 5: RAG BENCHMARKS & EVALUATION
# ---------------------------------------------------------
with tab5:
    st.subheader("📊 RAG Evaluation & Performance Benchmarks")
    st.caption("Quantitative metric comparisons across Vector-Only, BM25-Only, Hybrid RRF, and Hybrid + Rerank strategies.")

    eval_json_path = Path(__file__).resolve().parent / "data" / "evaluation_results.json"
    
    if eval_json_path.exists():
        with open(eval_json_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        summary_rows = []
        for mode, data in eval_data.items():
            summary_rows.append({
                "Retrieval Strategy": mode,
                "Precision@3": data["mean_precision_at_3"],
                "Recall@3": data["mean_recall_at_3"],
                "Groundedness Score": data["mean_groundedness_score"],
                "Avg Latency (s)": data["avg_latency_seconds"],
                "User Satisfaction (1-5)": data["mean_user_satisfaction"],
                "Hallucination Rate (%)": data["hallucination_rate_percent"]
            })
            
        df_summary = pd.DataFrame(summary_rows)
        st.table(df_summary)

        # Visual Metrics
        st.markdown("#### 📈 Strategy Performance Comparison")
        st.bar_chart(df_summary.set_index("Retrieval Strategy")[["Precision@3", "Recall@3", "Groundedness Score"]])
    else:
        st.info("ℹ️ Evaluation benchmark dataset is generating. Run `python src/evaluation.py` to populate realtime benchmarks.")

# ---------------------------------------------------------
# TAB 6: KNOWLEDGE BASE EXPLORER
# ---------------------------------------------------------
with tab6:
    st.subheader("📚 Policy Document Knowledge Base Explorer")
    st.caption("Inspect raw JSON policy wordings, clauses, and regulatory circulars stored in the dataset.")

    kb_files = list(DATA_DIR.glob("*.json"))
    if kb_files:
        selected_file = st.selectbox("Select Knowledge Base Dataset File", options=[f.name for f in kb_files])
        file_path = DATA_DIR / selected_file
        
        with open(file_path, "r", encoding="utf-8") as f:
            file_data = json.load(f)
            
        st.write(f"Loaded **{len(file_data)}** clauses from `{selected_file}`:")
        st.json(file_data)
    else:
        st.warning("No JSON files found in data/kb directory.")
