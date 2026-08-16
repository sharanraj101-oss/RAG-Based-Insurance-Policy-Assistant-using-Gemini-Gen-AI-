import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Ensure root directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import InsuranceAssistantPipeline
from src.preprocessing import Document

# Standard Test Dataset of Insurance Queries with Expected Ground Truth Clauses
TEST_DATASET = [
    {
        "id": "Q001",
        "query": "What is the waiting period for pre-existing diseases under health policy?",
        "expected_clauses": ["Section 4.1", "Section 4.2"],
        "category": "Health Insurance"
    },
    {
        "id": "Q002",
        "query": "What is covered under cashless hospitalization?",
        "expected_clauses": ["Section 3.1", "Section 3.2"],
        "category": "Health Insurance"
    },
    {
        "id": "Q003",
        "query": "How is zero depreciation coverage applied for motor insurance?",
        "expected_clauses": ["Section 2.1"],
        "category": "Motor Insurance"
    },
    {
        "id": "Q004",
        "query": "What is the claim settlement period specified in IRDAI guidelines?",
        "expected_clauses": ["IRDAI-REG-01", "IRDAI-REG-02"],
        "category": "Regulatory Compliance"
    },
    {
        "id": "Q005",
        "query": "Is suicide covered under life insurance policy?",
        "expected_clauses": ["Section 5.1"],
        "category": "Life Insurance"
    },
    {
        "id": "Q006",
        "query": "What are the exclusions for third-party liability in motor insurance?",
        "expected_clauses": ["Section 2.2", "Section 2.3"],
        "category": "Motor Insurance"
    },
    {
        "id": "Q007",
        "query": "Check claim status for CLM-1001",
        "expected_clauses": [],
        "expected_tool": "lookup_claim_status",
        "category": "Claims Support"
    },
    {
        "id": "Q008",
        "query": "Calculate premium for age 35 with sum insured 10 lakh health insurance",
        "expected_clauses": [],
        "expected_tool": "calculate_insurance_premium",
        "category": "Premium Calculator"
    }
]

class RAGEvaluator:
    def __init__(self, pipeline: InsuranceAssistantPipeline = None):
        self.pipeline = pipeline or InsuranceAssistantPipeline()

    def calculate_retrieval_metrics(self, retrieved_docs: List[Document], expected_clauses: List[str], top_k: int = 3) -> Tuple[float, float]:
        """
        Calculates Precision@K and Recall@K based on clause matching.
        """
        if not expected_clauses:
            return 1.0, 1.0  # N/A for tool-only queries
            
        retrieved_clauses = [doc.metadata.get("clause", "") for doc in retrieved_docs[:top_k]]
        
        # Check matching clauses
        hits = 0
        for exp in expected_clauses:
            if any(exp.lower() in r.lower() or r.lower() in exp.lower() for r in retrieved_clauses):
                hits += 1
                
        precision = hits / len(retrieved_clauses) if retrieved_clauses else 0.0
        recall = hits / len(expected_clauses) if expected_clauses else 0.0
        
        return round(precision, 3), round(recall, 3)

    def calculate_simulated_user_satisfaction(self, result: Dict[str, Any], precision: float, recall: float) -> float:
        """
        Simulates User Satisfaction Score (1.0 to 5.0 scale) based on latency, groundedness, citations, and precision.
        """
        score = 3.0  # Base score
        
        # Groundedness boost
        groundedness = result.get("groundedness_score", 0.0)
        score += groundedness * 1.0
        
        # Citation presence boost
        if result.get("citations"):
            score += 0.5
            
        # Tool usage boost
        if result.get("tool_used"):
            score += 0.5
            
        # Retrieval recall boost
        score += recall * 0.5
        
        # Latency penalty
        latency = result.get("latency_seconds", 0.0)
        if latency > 3.0:
            score -= 0.5
            
        return round(min(5.0, max(1.0, score)), 2)

    def run_benchmark(self, retrieval_modes: List[str] = None) -> Dict[str, Any]:
        """
        Evaluates the RAG pipeline across multiple retrieval strategies and outputs comprehensive benchmark report.
        """
        if retrieval_modes is None:
            retrieval_modes = ["vector_only", "bm25_only", "hybrid", "hybrid_rerank"]

        benchmark_summary = {}

        for mode in retrieval_modes:
            print(f"\n--- Running Evaluation Benchmark for Mode: [{mode}] ---")
            mode_results = []
            
            total_latency = 0.0
            total_precision = 0.0
            total_recall = 0.0
            total_groundedness = 0.0
            total_satisfaction = 0.0
            hallucination_count = 0
            
            for item in TEST_DATASET:
                res = self.pipeline.run_query(item["query"], retrieval_mode=mode, top_k=3)
                
                # Fetch retrieved docs
                if mode == "vector_only":
                    docs = [doc for doc, _ in self.pipeline.retriever_mgr.dense_search(item["query"], top_k=3)]
                elif mode == "bm25_only":
                    docs = [doc for doc, _ in self.pipeline.retriever_mgr.sparse_search(item["query"], top_k=3)]
                elif mode == "hybrid":
                    docs = self.pipeline.retriever_mgr.hybrid_search(item["query"], top_k=3)
                else:
                    candidates = self.pipeline.retriever_mgr.hybrid_search(item["query"], top_k=6)
                    reranked = self.pipeline.reranker.rerank(item["query"], candidates, top_k=3)
                    docs = [doc for doc, _ in reranked]
                    
                prec, rec = self.calculate_retrieval_metrics(docs, item.get("expected_clauses", []))
                satisfaction = self.calculate_simulated_user_satisfaction(res, prec, rec)
                
                is_hallucination = not res.get("is_grounded", True)
                if is_hallucination:
                    hallucination_count += 1
                    
                total_latency += res["latency_seconds"]
                total_precision += prec
                total_recall += rec
                total_groundedness += res["groundedness_score"]
                total_satisfaction += satisfaction
                
                mode_results.append({
                    "query_id": item["id"],
                    "query": item["query"],
                    "category": item["category"],
                    "precision_at_3": prec,
                    "recall_at_3": rec,
                    "groundedness_score": res["groundedness_score"],
                    "latency_seconds": res["latency_seconds"],
                    "satisfaction_score": satisfaction,
                    "tool_used": res["tool_used"],
                    "is_hallucination": is_hallucination
                })

            n = len(TEST_DATASET)
            summary_metrics = {
                "retrieval_mode": mode,
                "total_queries_tested": n,
                "avg_latency_seconds": round(total_latency / n, 4),
                "mean_precision_at_3": round(total_precision / n, 3),
                "mean_recall_at_3": round(total_recall / n, 3),
                "mean_groundedness_score": round(total_groundedness / n, 3),
                "mean_user_satisfaction": round(total_satisfaction / n, 2),
                "hallucination_rate_percent": round((hallucination_count / n) * 100, 1),
                "estimated_cost_per_1000_queries_usd": round(0.00015 * n * 1000 / n, 4),
                "detailed_evaluations": mode_results
            }
            
            benchmark_summary[mode] = summary_metrics

        return benchmark_summary

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    results = evaluator.run_benchmark()
    output_path = Path(__file__).resolve().parent.parent / "data" / "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Benchmark successfully finished! Saved results to {output_path}")
