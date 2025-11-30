"""
Preprocess knowledge base files for vector store.
"""
import os
from pathlib import Path
from typing import List


def load_knowledge_files(sources_dir: str = "sources") -> List[str]:
    """Load all text files from sources directory."""
    texts = []
    sources_path = Path(__file__).parent / sources_dir
    
    if not sources_path.exists():
        print(f"Warning: Sources directory {sources_path} does not exist")
        return texts
    
    for file_path in sources_path.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                texts.append({
                    'content': content,
                    'source': file_path.name,
                    'metadata': {'source_file': file_path.name}
                })
            print(f"Loaded: {file_path.name}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return texts


if __name__ == "__main__":
    texts = load_knowledge_files()
    print(f"Loaded {len(texts)} knowledge files")

