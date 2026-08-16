import pickle
import numpy as np
from typing import List, Tuple, Dict, Any
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from src.config import (
    VECTOR_STORE_DIR, 
    TOP_K_DENSE, 
    TOP_K_SPARSE, 
    DENSE_WEIGHT, 
    SPARSE_WEIGHT
)
from src.preprocessing import Document, load_policy_documents, create_document_chunks

class PolicyRetrieverManager:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        self.dense_matrix = None
        self.bm25 = None
        self.chunks: List[Document] = []
        self.tokenized_corpus = []
        self.doc_id_to_idx: Dict[int, int] = {}

    def build_index(self, force_rebuild: bool = False):
        """
        Builds both Dense TF-IDF matrix and BM25 Sparse index over policy chunks.
        """
        raw_docs = load_policy_documents()
        self.chunks = create_document_chunks(raw_docs)
        self.doc_id_to_idx = {id(doc): i for i, doc in enumerate(self.chunks)}
        
        corpus_texts = [doc.page_content for doc in self.chunks]
        
        # 1. Build Dense Vector Index (TF-IDF Cosine Embedding Space)
        print("Building Dense Vector Index...")
        self.dense_matrix = self.vectorizer.fit_transform(corpus_texts)
        
        # 2. Build Sparse BM25 Index
        print("Building Sparse BM25 Index...")
        self.tokenized_corpus = [text.lower().split() for text in corpus_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        print("Hybrid Retriever Index successfully built and ready!")

    def dense_search(self, query: str, top_k: int = TOP_K_DENSE) -> List[Tuple[Document, float]]:
        """Dense similarity search using vector embeddings."""
        if self.dense_matrix is None:
            raise ValueError("Index not built. Call build_index() first.")
            
        query_vec = self.vectorizer.transform([query])
        sim_scores = cosine_similarity(query_vec, self.dense_matrix).flatten()
        
        top_indices = np.argsort(sim_scores)[::-1][:top_k]
        return [(self.chunks[idx], float(sim_scores[idx])) for idx in top_indices if sim_scores[idx] > 0]

    def sparse_search(self, query: str, top_k: int = TOP_K_SPARSE) -> List[Tuple[Document, float]]:
        """Sparse keyword search using BM25Okapi."""
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_index() first.")
            
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        return [(self.chunks[idx], float(bm25_scores[idx])) for idx in top_indices if bm25_scores[idx] > 0]

    def hybrid_search(self, query: str, top_k: int = TOP_K_DENSE) -> List[Document]:
        """
        Combines dense vector search and sparse BM25 search via Reciprocal Rank Fusion (RRF).
        """
        dense_results = self.dense_search(query, top_k=top_k * 2)
        sparse_results = self.sparse_search(query, top_k=top_k * 2)
        
        rrf_scores: Dict[int, float] = {}
        
        # Reciprocal Rank Fusion formula: 1 / (rank + 60)
        for rank, (doc, score) in enumerate(dense_results):
            idx = self.doc_id_to_idx.get(id(doc))
            if idx is not None:
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (DENSE_WEIGHT / (rank + 60))
            
        for rank, (doc, score) in enumerate(sparse_results):
            idx = self.doc_id_to_idx.get(id(doc))
            if idx is not None:
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (SPARSE_WEIGHT / (rank + 60))
            
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
        return [self.chunks[idx] for idx in sorted_indices[:top_k]]

if __name__ == "__main__":
    mgr = PolicyRetrieverManager()
    mgr.build_index()
    
    query = "waiting period for pre-existing disease"
    results = mgr.hybrid_search(query, top_k=3)
    print(f"\n--- Hybrid Search Results for '{query}' ---")
    for doc in results:
        print(f"\n[{doc.metadata.get('category')} | Clause: {doc.metadata.get('clause')}]")
        print(doc.page_content)
