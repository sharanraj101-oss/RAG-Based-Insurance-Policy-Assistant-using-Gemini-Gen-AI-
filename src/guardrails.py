import re
from typing import Tuple, List, Dict, Any
from src.preprocessing import Document

class GuardrailsManager:
    def __init__(self):
        # Common PII Regex Patterns
        self.phone_pattern = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.aadhaar_pattern = re.compile(r'\b\d{4}\s?\d{4}\s?\d{4}\b')
        self.credit_card_pattern = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')

    def sanitize_input(self, text: str) -> Tuple[str, List[str]]:
        """
        Redacts PII (Personally Identifiable Information) from input user query.
        Returns (sanitized_text, list_of_redactions_made).
        """
        redactions = []
        sanitized = text
        
        if self.email_pattern.search(sanitized):
            sanitized = self.email_pattern.sub("[REDACTED_EMAIL]", sanitized)
            redactions.append("Email Address Redacted")
            
        if self.aadhaar_pattern.search(sanitized):
            sanitized = self.aadhaar_pattern.sub("[REDACTED_AADHAAR]", sanitized)
            redactions.append("Aadhaar Number Redacted")
            
        if self.credit_card_pattern.search(sanitized):
            sanitized = self.credit_card_pattern.sub("[REDACTED_CARD]", sanitized)
            redactions.append("Credit Card Number Redacted")
            
        if self.phone_pattern.search(sanitized):
            sanitized = self.phone_pattern.sub("[REDACTED_PHONE]", sanitized)
            redactions.append("Phone Number Redacted")
            
        return sanitized, redactions

    def check_groundedness(self, answer: str, context_docs: List[Document]) -> Tuple[bool, float]:
        """
        Checks if the generated answer is grounded in the retrieved context documents.
        Returns (is_grounded, confidence_score).
        """
        if not context_docs or not answer:
            return False, 0.0
            
        combined_context = " ".join([doc.page_content.lower() for doc in context_docs])
        answer_words = set(re.findall(r'\w+', answer.lower()))
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or', 'in', 'on', 'to', 'for', 'of', 'with', 'by', 'as', 'at', 'it', 'this', 'that', 'you', 'your'}
        content_words = answer_words - stop_words
        
        if not content_words:
            return True, 1.0
            
        matched_words = [w for w in content_words if w in combined_context]
        groundedness_ratio = len(matched_words) / len(content_words)
        
        is_grounded = groundedness_ratio >= 0.3
        return is_grounded, round(groundedness_ratio, 2)

if __name__ == "__main__":
    guard = GuardrailsManager()
    sample_text = "My phone is 9876543210 and email is test@domain.com. How do I file a claim?"
    clean, red = guard.sanitize_input(sample_text)
    print(f"Sanitized: {clean}")
    print(f"Redactions: {red}")
