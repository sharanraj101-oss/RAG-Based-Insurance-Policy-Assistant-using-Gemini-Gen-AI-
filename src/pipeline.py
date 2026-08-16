import os
import sys
import time
import re
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Ensure root directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY, LLM_MODEL_NAME, SYSTEM_PROMPT
from src.retriever import PolicyRetrieverManager
from src.reranker import PolicyReranker
from src.agent_tools import lookup_claim_status, calculate_insurance_premium
from src.guardrails import GuardrailsManager
from src.preprocessing import Document

class InsuranceAssistantPipeline:
    def __init__(self, mode: str = "hybrid_rerank"):
        self.mode = mode
        print("Initializing Policy Assistant Pipeline...")
        self.retriever_mgr = PolicyRetrieverManager()
        self.retriever_mgr.build_index()
        self.reranker = PolicyReranker()
        self.guardrails = GuardrailsManager()
        
        # Initialize Gemini Client if API key is configured
        self.api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.client = None
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                print("Gemini API Client successfully initialized!")
            except Exception as e:
                print(f"Warning: Gemini API Client init failed ({e}). Fallback logic will be active.")

    def run_query(self, user_query: str, retrieval_mode: str = "hybrid_rerank", top_k: int = 3) -> Dict[str, Any]:
        """
        Executes end-to-end RAG query workflow.
        Returns detailed result dictionary including response text, sources, tool outputs, latency, and guardrails log.
        """
        start_time = time.time()
        
        # 1. Input Guardrails & PII Sanitization
        sanitized_query, redactions = self.guardrails.sanitize_input(user_query)
        
        # 2. Check for Agentic Tool Invocation
        tool_output = None
        tool_used = None
        
        # Claim ID Check (e.g. CLM-1001)
        if "clm-" in sanitized_query.lower() or "claim status" in sanitized_query.lower():
            words = sanitized_query.replace(":", " ").replace(",", " ").split()
            for word in words:
                if word.upper().startswith("CLM-"):
                    tool_used = "lookup_claim_status"
                    tool_output = lookup_claim_status(word.upper())
                    break

        # Premium Calculator Check (e.g. "calculate premium", "premium for age 35")
        if not tool_used and any(k in sanitized_query.lower() for k in ["calculate premium", "premium for", "estimate premium", "premium quote"]):
            tool_used = "calculate_insurance_premium"
            # Extract age
            age_match = re.search(r'\bage\s*(\d{1,2})\b', sanitized_query, re.IGNORECASE)
            age = int(age_match.group(1)) if age_match else 35
            
            # Extract sum insured (e.g. 1000000, 10 lakh, 5 lakh)
            sum_insured = 500000.0
            if "10 lakh" in sanitized_query.lower() or "1,000,000" in sanitized_query or "1000000" in sanitized_query:
                sum_insured = 1000000.0
            elif "5 lakh" in sanitized_query.lower() or "500,000" in sanitized_query or "500000" in sanitized_query:
                sum_insured = 500000.0
            elif "20 lakh" in sanitized_query.lower() or "2,000,000" in sanitized_query or "2000000" in sanitized_query:
                sum_insured = 2000000.0
            
            plan_type = "motor" if "motor" in sanitized_query.lower() or "car" in sanitized_query.lower() else "health"
            zero_dep = "zero dep" in sanitized_query.lower() or "add-on" in sanitized_query.lower()
            
            tool_output = calculate_insurance_premium(age=age, sum_insured=sum_insured, plan_type=plan_type, zero_dep=zero_dep)
                    
        # 3. Policy Document Retrieval based on selected mode
        retrieved_docs: List[Document] = []
        if retrieval_mode == "vector_only":
            raw_results = self.retriever_mgr.dense_search(sanitized_query, top_k=top_k)
            retrieved_docs = [doc for doc, _ in raw_results]
        elif retrieval_mode == "bm25_only":
            raw_results = self.retriever_mgr.sparse_search(sanitized_query, top_k=top_k)
            retrieved_docs = [doc for doc, _ in raw_results]
        elif retrieval_mode == "hybrid":
            retrieved_docs = self.retriever_mgr.hybrid_search(sanitized_query, top_k=top_k)
        else:  # hybrid_rerank (Default Best Mode)
            candidates = self.retriever_mgr.hybrid_search(sanitized_query, top_k=top_k * 2)
            reranked_pairs = self.reranker.rerank(sanitized_query, candidates, top_k=top_k)
            retrieved_docs = [doc for doc, _ in reranked_pairs]

        # 4. Context Preparation
        context_str = ""
        citations = []
        for i, doc in enumerate(retrieved_docs):
            source_info = f"Source {i+1} [{doc.metadata.get('source')} | {doc.metadata.get('clause')} - {doc.metadata.get('title')}]:"
            context_str += f"{source_info}\n{doc.page_content}\n\n"
            citations.append({
                "source": doc.metadata.get("source"),
                "clause": doc.metadata.get("clause"),
                "title": doc.metadata.get("title"),
                "category": doc.metadata.get("category"),
                "snippet": doc.page_content[:180] + "..."
            })

        # 5. Gemini LLM Synthesis Prompt
        prompt = f"""System Instructions:
{SYSTEM_PROMPT}

User Query:
{sanitized_query}

Tool Status / Output:
{tool_output if tool_output else "No tool required for this query."}

Retrieved Policy Documents Context:
{context_str if context_str else "No relevant policy documents found."}

Please generate a helpful, accurate, grounded answer addressing the user's query:"""

        # 6. Response Generation
        answer_text = ""
        if self.client:
            try:
                response = self.client.models.generate_content(
                    model=LLM_MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=800
                    )
                )
                answer_text = response.text
            except Exception as e:
                print(f"Gemini API generation error: {e}")
                answer_text = self._fallback_response_generator(sanitized_query, tool_output, retrieved_docs)
        else:
            answer_text = self._fallback_response_generator(sanitized_query, tool_output, retrieved_docs)

        # 7. Output Guardrails & Groundedness Checking
        is_grounded, groundedness_score = self.guardrails.check_groundedness(answer_text, retrieved_docs)
        
        latency = round(time.time() - start_time, 3)

        return {
            "query": user_query,
            "sanitized_query": sanitized_query,
            "answer": answer_text,
            "tool_used": tool_used,
            "tool_output": tool_output,
            "citations": citations,
            "retrieval_mode": retrieval_mode,
            "retrieved_count": len(retrieved_docs),
            "redactions": redactions,
            "is_grounded": is_grounded,
            "groundedness_score": groundedness_score,
            "latency_seconds": latency
        }

    def _fallback_response_generator(self, query: str, tool_output: Any, docs: List[Document]) -> str:
        """Generates grounded answer structure when API key is offline or in test mode."""
        lines = []
        
        if tool_output:
            lines.append("### 🔍 Live Tool Response")
            lines.append(f"```json\n{tool_output}\n```\n")
            
        if docs:
            lines.append("### 📋 Policy Details & Coverage")
            for doc in docs:
                lines.append(f"**[{doc.metadata.get('category')} - {doc.metadata.get('clause')}: {doc.metadata.get('title')}]**")
                lines.append(f"{doc.page_content.split('Details: ')[-1]}\n")
            lines.append("---")
            lines.append("*Note: Grounded directly from official policy clause records.*")
        else:
            lines.append("I could not find relevant details in the current policy documentation.")
            
        return "\n".join(lines)

if __name__ == "__main__":
    pipeline = InsuranceAssistantPipeline()
    res = pipeline.run_query("What is the waiting period for pre-existing disease?", retrieval_mode="hybrid_rerank")
    print("\n--- Pipeline Execution Output ---")
    print(f"Latency: {res['latency_seconds']}s | Groundedness: {res['groundedness_score']}")
    print(f"Answer:\n{res['answer']}".encode('ascii', errors='ignore').decode('ascii'))
