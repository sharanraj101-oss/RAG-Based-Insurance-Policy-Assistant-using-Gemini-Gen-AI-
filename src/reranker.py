from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import Document
from src.config import TOP_K_FINAL

class PolicyReranker:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3))

    def rerank(self, query: str, documents: List[Document], top_k: int = TOP_K_FINAL) -> List[Tuple[Document, float]]:
        """
        Re-ranks candidate document chunks against the query using character/n-gram cross-matching.
        Returns (Document, score) tuples sorted descending by relevance score.
        """
        if not documents:
            return []
            
        corpus = [doc.page_content for doc in documents]
        all_texts = [query] + corpus
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            query_vec = tfidf_matrix[0]
            doc_vecs = tfidf_matrix[1:]
            
            sim_scores = cosine_similarity(query_vec, doc_vecs).flatten()
            
            doc_scores = []
            for doc, score in zip(documents, sim_scores):
                # Apply clause number boost if query mentions specific clause numbers
                boost = 0.0
                clause = doc.metadata.get("clause", "").lower()
                if clause and clause in query.lower():
                    boost += 0.2
                final_score = float(score) + boost
                doc_scores.append((doc, final_score))
                
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            return doc_scores[:top_k]
        except Exception as e:
            # Fallback if vectorizer encounters unexpected formatting
            return [(doc, 1.0 / (i + 1)) for i, doc in enumerate(documents[:top_k])]

if __name__ == "__main__":
    from src.preprocessing import load_policy_documents
    docs = load_policy_documents()
    reranker = PolicyReranker()
    results = reranker.rerank("30-day settlement rule", docs[:5], top_k=2)
    for doc, score in results:
        print(f"Rerank Score: {score:.4f} | Title: {doc.metadata.get('title')}")
