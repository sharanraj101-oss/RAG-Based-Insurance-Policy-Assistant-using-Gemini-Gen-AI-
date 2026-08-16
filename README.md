# 🛡️ RAG-Based Insurance Policy Assistant using Gemini (Gen AI)

A state-of-the-art Retrieval-Augmented Generation (RAG) system built with **Google Gemini LLM**, **Hybrid Search (Dense + BM25)**, **Cross-Encoder Reranking**, **Agentic Tool Calling**, and **Guardrails Safety Layers** for plain-language, grounded insurance policy assistance.

---

## 🌟 Key Features

- **📖 Clause-Grounded Policy Q&A**: Answers queries across Health, Motor, Life insurance policies and IRDAI regulatory guidelines with explicit clause citation.
- **🔀 Hybrid RRF & Cross-Encoder Retrieval**: Combines Dense Semantic Search (TF-IDF Cosine Space) and Sparse Keyword Search (BM25Okapi) using Reciprocal Rank Fusion (RRF), followed by Cross-Encoder scoring for maximum precision.
- **🤖 Agentic Tool Integration**:
  - **Live Claim Status Tracker**: Recognizes Claim IDs (e.g., `CLM-1001`, `CLM-1002`, `CLM-1003`) and retrieves real-time claim status, approval amounts, and surveyor notes.
  - **Dynamic Premium Calculator**: Computes estimated annual premiums, rider add-ons, and 18% GST based on age and coverage amount.
- **🛡️ Guardrails & Safety Layer**:
  - **PII Redaction**: Automatically redacts sensitive user information (Phone Numbers, Emails, Aadhaar Numbers, Credit Card Numbers) before LLM prompt submission.
  - **Groundedness & Hallucination Guard**: Verifies whether generated responses are strictly supported by retrieved policy clauses.
- **📊 Quantitative Evaluation & Benchmarking**: Evaluates Precision@K, Recall@K, Groundedness Score, Latency, User Satisfaction Score, Hallucination Rate, and Cost per Query across retrieval modes.
- **💎 Modern Streamlit Interface**: Dark glassmorphism interactive dashboard with multi-tab navigation, quick test prompts, metrics visualization, and knowledge base explorer.

---

## 🏗️ System Architecture

```
                               ┌─────────────────────────┐
                               │   User Input Query      │
                               └────────────┬────────────┘
                                            │
                               ┌────────────▼────────────┐
                               │  Guardrails Manager     │ (PII Redaction & Sanitization)
                               └────────────┬────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    │                                               │
       [Agentic Tool Detected?]                            [Policy Query]
                    │                                               │
       ┌────────────▼────────────┐                   ┌──────────────▼──────────────┐
       │  Claim / Premium Tool   │                   │   Hybrid Retrieval Engine   │
       └────────────┬────────────┘                   │  • Dense Vector Search      │
                    │                                │  • Sparse BM25 Search       │
                    │                                └──────────────┬──────────────┘
                    │                                               │
                    │                                ┌──────────────▼──────────────┐
                    │                                │  Reciprocal Rank Fusion     │
                    │                                └──────────────┬──────────────┘
                    │                                               │
                    │                                ┌──────────────▼──────────────┐
                    │                                │   Cross-Encoder Reranker    │
                    │                                └──────────────┬──────────────┘
                    │                                               │
                    └───────────────────────┬───────────────────────┘
                                            │
                               ┌────────────▼────────────┐
                               │  Gemini LLM Synthesizer │
                               └────────────┬────────────┘
                                            │
                               ┌────────────▼────────────┐
                               │ Groundedness Validator  │
                               └────────────┬────────────┘
                                            │
                               ┌────────────▼────────────┐
                               │ Streamlit User Interface│
                               └─────────────────────────┘
```

---

## 📁 Project Directory Structure

```
C:/Final Project/
├── app.py                      # Interactive Streamlit Web Application
├── README.md                   # Comprehensive Project Overview & Guide
├── ARCHITECTURE.md             # In-depth System Architecture & Technical Specifications
├── EVALUATION_REPORT.md        # Quantitative Evaluation & Performance Benchmark Report
├── Project 1 - Gen_AI_Project_(with added complexity).docx.pdf  # Project Specs
├── data/
│   ├── kb/                     # Policy Knowledge Base JSON Datasets
│   │   ├── health_policy.json
│   │   ├── motor_policy.json
│   │   ├── life_policy.json
│   │   └── irdai_guidelines.json
│   └── evaluation_results.json # Generated Evaluation Metrics Dataset
└── src/
    ├── __init__.py
    ├── config.py               # System Configurations & Parameters
    ├── preprocessing.py        # Data Ingestion & Clause-Level Chunking
    ├── retriever.py            # Dense + Sparse Hybrid RRF Search
    ├── reranker.py             # Cross-Encoder Clause Reranking Engine
    ├── agent_tools.py          # Claim Status & Premium Calculator Tools
    ├── guardrails.py           # PII Redaction & Hallucination Prevention
    ├── pipeline.py             # End-to-End RAG Execution Pipeline
    └── evaluation.py           # RAG Evaluation Suite (Precision, Recall, RAGAS)
```

---

## 🚀 Quick Start Guide

### 1. Requirements & Prerequisites
- Python 3.9+
- Dependencies: `streamlit`, `google-genai`, `scikit-learn`, `rank-bm25`, `pandas`, `numpy`, `python-dotenv`

### 2. Configure Environment Variables
Create a `.env` file in the project root directory or set system environment variables:
```env
GEMINI_API_KEY=your_gemini_api_key_here
LLM_MODEL_NAME=gemini-2.5-flash
```

### 3. Run the Demonstration Web Application
Start the Streamlit interface:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Execute Benchmark Evaluation Suite
To re-run the benchmark evaluation across all retrieval strategies (Vector-Only, BM25-Only, Hybrid RRF, Hybrid + Rerank):
```bash
python src/evaluation.py
```

---

## 📊 Benchmark Evaluation Summary

| Retrieval Strategy | Precision@3 | Recall@3 | Groundedness Score | Avg Latency (s) | User Satisfaction (1-5) | Hallucination Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid + Rerank (Recommended)** | **0.875** | **0.950** | **0.982** | 0.015s | **4.78** | **0.0%** |
| **Hybrid RRF** | 0.792 | 0.900 | 0.945 | 0.012s | 4.52 | 0.0% |
| **Dense Vector Only** | 0.708 | 0.825 | 0.890 | 0.008s | 4.21 | 12.5% |
| **BM25 Sparse Only** | 0.625 | 0.750 | 0.835 | 0.006s | 3.94 | 25.0% |

---

## 📄 License & Attribution
Developed for the GUVI / HCL Gen AI Capstone Project on RAG-Based Insurance Policy Assistance.
