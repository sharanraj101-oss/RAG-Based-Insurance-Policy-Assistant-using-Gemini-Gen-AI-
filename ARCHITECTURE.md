# 🏗️ Technical Architecture & System Design Document

## 1. Overview
The **RAG-Based Insurance Policy Assistant** is a enterprise-grade Retrieval-Augmented Generation system designed to answer complex policy questions, perform real-time claim status lookups, calculate premium estimates, and enforce strict safety guardrails.

---

## 2. Architecture Component Breakdown

```
                   ┌────────────────────────────────────────┐
                   │           User Query Input             │
                   └───────────────────┬────────────────────┘
                                       │
                   ┌───────────────────▼────────────────────┐
                   │    1. Guardrails & PII Redaction       │
                   └───────────────────┬────────────────────┘
                                       │
         ┌─────────────────────────────┴─────────────────────────────┐
         │                                                           │
┌────────▼─────────────────────────────────┐   ┌─────────────────────▼─────────────────────┐
│ 2. Agentic Tool Router                    │   │ 3. Policy Document Retriever Engine       │
│  • Claim Lookup (`CLM-XXXX`)             │   │  • Dense Vector Search (TF-IDF Cosine)    │
│  • Premium Calculation (Age/Coverage)    │   │  • Sparse Search (BM25Okapi)              │
└────────┬─────────────────────────────────┘   └─────────────────────┬─────────────────────┘
         │                                                           │
         │                                     ┌─────────────────────▼─────────────────────┐
         │                                     │ 4. Reciprocal Rank Fusion (RRF)           │
         │                                     └─────────────────────┬─────────────────────┘
         │                                                           │
         │                                     ┌─────────────────────▼─────────────────────┐
         │                                     │ 5. Cross-Encoder Clause Reranker          │
         │                                     └─────────────────────┬─────────────────────┘
         │                                                           │
         └─────────────────────────────┬─────────────────────────────┘
                                       │
                   ┌───────────────────▼────────────────────┐
                   │ 6. Context Assembler & System Prompt    │
                   └───────────────────┬────────────────────┘
                                       │
                   ┌───────────────────▼────────────────────┐
                   │ 7. Gemini LLM Generation Engine        │
                   └───────────────────┬────────────────────┘
                                       │
                   ┌───────────────────▼────────────────────┐
                   │ 8. Output Groundedness Validator       │
                   └───────────────────┬────────────────────┘
                                       │
                   ┌───────────────────▼────────────────────┐
                   │ 9. Streamlit Interactive Web Interface │
                   └────────────────────────────────────────┘
```

---

## 3. Core Modules & Implementation Details

### 3.1 Data Preprocessing & Document Chunking (`src/preprocessing.py`)
- **Knowledge Base Source**: Structured JSON documents containing policy titles, categories, clauses, and content body (`health_policy.json`, `motor_policy.json`, `life_policy.json`, `irdai_guidelines.json`).
- **Chunking Strategy**: Adaptive boundary splitter configured with `CHUNK_SIZE = 400` characters and `CHUNK_OVERLAP = 50` characters. Respects paragraph breaks (`\n`) and sentence endings (`. `) to ensure clause integrity.

### 3.2 Hybrid Retrieval Subsystem (`src/retriever.py`)
- **Dense Vector Search**: Fits a TF-IDF bi-gram vectorizer across document corpus to build a dense semantic space, scoring candidates via Cosine Similarity.
- **Sparse Keyword Search**: Implements `BM25Okapi` over tokenized lowercased corpus texts to capture exact legal terms and clause numbers.
- **Reciprocal Rank Fusion (RRF)**: Combines dense and sparse search scores:
  $$RRF\_Score(doc) = \frac{w_{dense}}{rank_{dense} + 60} + \frac{w_{sparse}}{rank_{sparse} + 60}$$
  where $w_{dense} = 0.6$ and $w_{sparse} = 0.4$.

### 3.3 Cross-Encoder Clause Reranker (`src/reranker.py`)
- Re-scores candidate retrieved chunks against the user query using high-order n-gram cross-matching.
- Applies dynamic score boosts when specific clause references (e.g. `Section 4.1`, `IRDAI-REG-01`) are present in both query and metadata.

### 3.4 Agentic Tool Calling (`src/agent_tools.py`)
- **`lookup_claim_status(claim_id)`**: Queries live claim database returning claimant details, policy numbers, claim status (`APPROVED`, `UNDER_INSPECTION`, `DOCUMENTS_REQUIRED`), approved amounts, and surveyor notes.
- **`calculate_insurance_premium(age, sum_insured, plan_type, zero_dep)`**: Calculates base premium modified by age risk tiers (<30: 0.9x, 30-44: 1.1x, 45-59: 1.4x, 60+: 1.8x), rider costs, and 18% mandatory GST.

### 3.5 Guardrails & Safety Inspector (`src/guardrails.py`)
- **Input Sanitization**: Scans incoming text using optimized Regular Expressions for Phone Numbers, Email Addresses, Aadhaar Numbers (12-digit Indian national ID), and Credit Card Numbers, replacing them with `[REDACTED_*]` tokens.
- **Groundedness Verification**: Measures the content-word overlap ratio between the synthesized answer and retrieved policy contexts, flagging ungrounded responses.

### 3.6 Gemini LLM Synthesis (`src/pipeline.py`)
- Uses Google Gemini LLM (`gemini-2.5-flash`) via the official `google-genai` SDK with controlled temperature ($T = 0.2$) and strict system prompts preventing hallucinated coverage claims.

---

## 4. Frontend & User Interface Design (`app.py`)
Built with Streamlit utilizing modern CSS glassmorphism, responsive control sidebars, multi-tab routing, quick test prompt buttons, claim trackers, dynamic premium quotes, guardrail inspectors, and live evaluation charts.
