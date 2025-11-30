# MedInsight Architecture

## System Overview

MedInsight is a full-stack AI-powered medical audit system that analyzes clinical notes, prescriptions, and medical documents to identify potential safety issues, drug interactions, and guideline compliance.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                       │
│  (Upload, Visualization, Reports, Analytics)                │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP/REST
┌──────────────────────▼──────────────────────────────────────┐
│                    FastAPI Backend                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ /ingest  │  │  /audit  │  │ /health  │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼──────┐ ┌────▼─────┐ ┌─────▼──────┐
│   Ingestion  │ │    NER   │ │ Normalizer │
│   Service    │ │  Service │ │  Service   │
└───────┬──────┘ └────┬─────┘ └─────┬──────┘
        │             │              │
        └─────────────┼──────────────┘
                      │
        ┌─────────────┼──────────────┐
        │             │              │
┌───────▼──────┐ ┌───▼────┐ ┌───────▼────────┐
│  RAG Service │ │ Agents │ │ Decision Engine│
│  (Chroma)    │ │        │ │                │
└───────┬──────┘ └───┬────┘ └───────┬────────┘
        │            │              │
        └────────────┼──────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
┌───────▼────┐ ┌─────▼─────┐ ┌───▼──────┐
│  Chroma    │ │  MongoDB  │ │ Knowledge│
│  Vector DB │ │           │ │   Base   │
└────────────┘ └───────────┘ └──────────┘
```

## Components

### 1. Ingestion Service
- **Purpose**: Extract text from PDFs and images
- **Technologies**: pdfplumber, pytesseract
- **Output**: Raw text and chunked text

### 2. NER Service
- **Purpose**: Extract medical entities (drugs, dosages, symptoms, etc.)
- **Technologies**: HuggingFace transformers, rule-based patterns
- **Output**: Structured entities with types and positions

### 3. Normalizer Service
- **Purpose**: Normalize entities to standard formats
- **Functions**:
  - Convert dosages to mg
  - Standardize frequencies (qd/bid/tid → standardized)
  - Normalize lab units
- **Output**: Normalized entity dictionary

### 4. RAG Service
- **Purpose**: Retrieve relevant medical knowledge
- **Technologies**: ChromaDB, LangChain, sentence-transformers
- **Embeddings**: all-mpnet-base-v2
- **Output**: Relevant document chunks with metadata

### 5. Agent System
Five specialized agents:

1. **DosageCheckerAgent**: Validates medication dosages
2. **InteractionCheckerAgent**: Detects drug interactions
3. **RedFlagCheckerAgent**: Identifies emergency symptoms
4. **MissingTestsCheckerAgent**: Checks for required lab tests
5. **GuidelineComplianceCheckerAgent**: Validates treatment guidelines

### 6. Decision Engine
- **Purpose**: Merge agent outputs into final report
- **Functions**:
  - Categorize issues by severity
  - Calculate risk score (0-10)
  - Generate recommendations
- **Output**: Complete audit report

### 7. Data Storage
- **MongoDB**: Stores audit results and history
- **ChromaDB**: Vector database for knowledge base

## Data Flow

1. **Upload**: User uploads PDF/image via Streamlit
2. **Ingestion**: Extract text using OCR/PDF parsing
3. **NER**: Extract medical entities
4. **Normalization**: Standardize entity formats
5. **RAG Retrieval**: Query knowledge base for relevant information
6. **Agent Analysis**: Each agent analyzes specific aspects
7. **Decision Engine**: Merge results and generate report
8. **Storage**: Save to MongoDB
9. **Display**: Show results in Streamlit dashboard

## Technology Stack

- **Backend**: FastAPI, Python
- **Frontend**: Streamlit
- **NLP**: HuggingFace Transformers, LangChain
- **Vector DB**: ChromaDB
- **Database**: MongoDB
- **OCR**: pytesseract
- **PDF**: pdfplumber
- **Embeddings**: sentence-transformers

## Scalability Considerations

- Vector store can be scaled horizontally
- MongoDB supports sharding for large datasets
- FastAPI supports async operations
- Agents can be parallelized
- RAG retrieval is optimized with MMR

