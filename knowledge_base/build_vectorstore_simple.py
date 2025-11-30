#!/usr/bin/env python3
"""
Simple vector store builder that works with current dependencies.
"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document
    except ImportError:
        print("Error: Please install: pip install langchain-community langchain-text-splitters sentence-transformers")
        sys.exit(1)

from preprocess import load_knowledge_files

def main():
    print("🏗️  Building MedInsight Vector Store")
    print("=" * 50)
    
    # Load knowledge files
    print("\n📚 Loading knowledge files...")
    knowledge_data = load_knowledge_files()
    
    if not knowledge_data:
        print("❌ No knowledge files found!")
        return
    
    print(f"✅ Loaded {len(knowledge_data)} knowledge files")
    
    # Initialize embeddings
    print("\n🔤 Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    print("✅ Embeddings ready")
    
    # Split into chunks: 300 tokens per chunk, 30 token overlap
    # Approximate: 1 token ≈ 4 characters, so 300 tokens ≈ 1200 chars, 30 tokens ≈ 120 chars
    print("\n✂️  Splitting documents into chunks (300 tokens, 30 overlap)...")
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
    
    print(f"✅ Created {len(documents)} document chunks")
    
    # Create vector store
    print("\n💾 Creating vector store...")
    persist_dir = Path(__file__).parent.parent / "chroma_db"
    persist_dir.mkdir(exist_ok=True)
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_dir)
    )
    
    # Persist
    vectorstore.persist()
    print(f"✅ Vector store saved to {persist_dir}")
    
    # Test retrieval
    print("\n🧪 Testing retrieval...")
    results = vectorstore.similarity_search("aspirin dosage", k=2)
    print(f"✅ Retrieved {len(results)} test documents")
    
    print("\n" + "=" * 50)
    print("🎉 Vector store build complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()

