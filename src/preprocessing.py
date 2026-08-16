import json
from typing import List, Dict, Any
from pathlib import Path
from src.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

class Document:
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Document(id={self.metadata.get('id')}, title='{self.metadata.get('title')}')"

def load_policy_documents(data_dir: Path = DATA_DIR) -> List[Document]:
    """
    Loads all JSON policy files from the data directory and converts them into Document objects.
    """
    documents = []
    json_files = list(data_dir.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                records = json.load(f)
                
            for record in records:
                page_content = f"Title: {record.get('title', '')}\n" \
                               f"Category: {record.get('category', '')}\n" \
                               f"Clause: {record.get('clause', '')}\n" \
                               f"Details: {record.get('content', '')}"
                
                metadata = {
                    "id": record.get("id", ""),
                    "title": record.get("title", ""),
                    "category": record.get("category", ""),
                    "clause": record.get("clause", ""),
                    "source": json_file.name
                }
                
                documents.append(Document(page_content=page_content, metadata=metadata))
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")
            
    print(f"Loaded {len(documents)} raw policy records from {len(json_files)} datasets.")
    return documents

def create_document_chunks(documents: List[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """
    Splits policy documents into retrievable units preserving clause context.
    Ensures robust non-blocking forward sliding window.
    """
    chunks = []
    min_step = max(1, chunk_size - chunk_overlap)
    
    for doc in documents:
        text = doc.page_content
        if len(text) <= chunk_size:
            chunks.append(doc)
            continue
            
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                search_start = start + max(10, chunk_overlap)
                break_point = text.rfind("\n", search_start, end)
                if break_point == -1:
                    break_point = text.rfind(". ", search_start, end)
                if break_point > search_start:
                    end = break_point + 1
                    
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Document(page_content=chunk_text, metadata=doc.metadata.copy()))
            
            if end >= len(text):
                break
                
            next_start = max(start + min_step, end - chunk_overlap)
            if next_start <= start:
                next_start = start + min_step
            start = next_start
            
    print(f"Created {len(chunks)} retrievable text chunks.")
    return chunks

if __name__ == "__main__":
    docs = load_policy_documents()
    chunks = create_document_chunks(docs)
    print(f"Sample Chunk:\n{chunks[0].page_content}\nMetadata: {chunks[0].metadata}")
