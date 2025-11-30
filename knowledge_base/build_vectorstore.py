"""
Build Chroma vector store from knowledge base files.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.schema import Document
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document
    except ImportError:
        try:
            from langchain.vectorstores import Chroma
            from langchain.embeddings import HuggingFaceEmbeddings
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain.schema import Document
        except ImportError:
            print("Error: langchain not installed. Please install: pip install langchain langchain-community langchain-text-splitters")
            exit(1)

from preprocess import load_knowledge_files


def build_vectorstore(persist_directory: str = "../chroma_db"):
    """Build and persist vector store from knowledge base."""
    print("Building vector store...")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    # Load knowledge files
    print("Loading knowledge files...")
    knowledge_data = load_knowledge_files()
    
    if not knowledge_data:
        print("No knowledge files found. Please add .txt files to sources/ directory.")
        return
    
    # Split texts into chunks: 300 tokens per chunk, 30 token overlap
    # Approximate: 1 token ≈ 4 characters, so 300 tokens ≈ 1200 chars, 30 tokens ≈ 120 chars
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # ~300 tokens (300 * 4 chars)
        chunk_overlap=120,  # ~30 tokens (30 * 4 chars)
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Better semantic chunking
    )
    
    documents = []
    for item in knowledge_data:
        chunks = text_splitter.split_text(item['content'])
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    'source': item['source'],
                    **item.get('metadata', {})
                }
            )
            documents.append(doc)
    
    print(f"Created {len(documents)} document chunks")
    
    # Create vector store
    print("Creating vector store...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Persist
    vectorstore.persist()
    print(f"Vector store saved to {persist_directory}")
    print("Vector store build complete!")


if __name__ == "__main__":
    # Get persist directory from command line or use default
    persist_dir = sys.argv[1] if len(sys.argv) > 1 else "./chroma_db"
    build_vectorstore(persist_dir)

