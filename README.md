# MedInsight

A comprehensive clinical notes auditing system for automated medical safety analysis and risk assessment.

## Overview

MedInsight is a production-ready system that analyzes clinical notes, prescriptions, and medical documents to identify potential safety issues, drug interactions, dosage errors, and guideline compliance violations. Built with modern AI technologies including NLP, RAG (Retrieval-Augmented Generation), and a multi-agent architecture, MedInsight provides comprehensive medical audit reports with actionable insights.

## Features

- **Document Processing**: Extract text from PDFs and images using OCR
- **Medical Entity Recognition**: Automatically extract drugs, dosages, symptoms, vitals, and lab values
- **RAG-Powered Analysis**: Retrieve relevant medical guidelines from knowledge base
- **Multi-Agent System**: Five specialized agents for dosage safety, drug interactions, red flags, missing tests, and guideline compliance
- **Risk Scoring**: Automated risk assessment with severity classification (Critical, High, Moderate, Low, Safe)
- **Analytics Dashboard**: Real-time analytics with KPIs, trends, and clinical category analysis
- **RESTful API**: FastAPI backend with comprehensive endpoints
- **MongoDB Integration**: Persistent storage for audit history

## Technology Stack

- **Backend**: Python 3.8+, FastAPI, Uvicorn
- **AI/ML**: LangChain, Hugging Face Transformers, OpenAI GPT-4
- **NLP**: BERT-based Named Entity Recognition, sentence-transformers for embeddings
- **Vector Database**: Chroma DB for semantic search and RAG
- **Database**: MongoDB for persistent audit storage
- **Frontend**: Streamlit with Plotly for interactive visualizations
- **Document Processing**: pdfplumber for PDF extraction, pytesseract for OCR

## Project Structure

```
MedInsight/
├── backend/              # FastAPI application
│   ├── app/
│   │   ├── main.py      # FastAPI entry point
│   │   ├── routes/      # API endpoints
│   │   ├── services/    # Core services (NER, RAG, Agents, Decision Engine)
│   │   └── models/     # Data models and database
│   └── tests/          # Test cases
├── dashboard/           # Streamlit dashboard
│   ├── streamlit_app.py
│   ├── components/     # UI components
│   └── pages/         # Dashboard pages
├── knowledge_base/     # Medical knowledge sources and vector store
│   ├── sources/       # Knowledge files (.txt)
│   └── build_vectorstore.py
└── docs/              # Architecture and design documentation
```

## Installation

### Prerequisites

- Python 3.8 or higher
- MongoDB (local or remote)
- Tesseract OCR (for image processing)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd MedInsight
```

2. **Create virtual environment**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up MongoDB**
```bash
# macOS (Homebrew)
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community

# Or use existing MongoDB instance
# Update MONGODB_URI in .env if needed
```

5. **Build vector store**
```bash
cd ../knowledge_base
source ../backend/venv/bin/activate
python build_vectorstore.py
```

6. **Configure environment variables** (optional)
```bash
cd ../backend
# Create .env file
echo "MONGODB_URI=mongodb://localhost:27017/" > .env
echo "OPENAI_API_KEY=your_key_here" >> .env
```

## Running the Application

### Quick Start (Both Services)

**Easiest way - Start both backend and dashboard:**
```bash
./start.sh
```

This will:
- Check if ports are available (stops existing services if needed)
- Start MongoDB (if not running)
- Start FastAPI backend on `http://localhost:8000`
- Start Streamlit dashboard on `http://localhost:8501`

**To stop all services:**
```bash
./stop.sh
```

### Manual Start (Separate Terminals)

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/api/health`

**Terminal 2 - Dashboard:**
```bash
cd dashboard
source ../backend/venv/bin/activate
streamlit run streamlit_app.py
```

The dashboard will be available at `http://localhost:8501`

### Alternative: Use Individual Scripts

```bash
# Terminal 1
./start_backend.sh

# Terminal 2
./start_dashboard.sh
```

## Usage

### Using the Dashboard

1. Navigate to the "Upload & Audit" page
2. Upload a PDF/image or enter clinical text
3. Click "Run Complete Audit"
4. View results on the "View Results" page
5. Access analytics on the "Analytics" page

### Using the API

#### Health Check
```bash
curl http://localhost:8000/api/health
```

#### Run Audit (Text Input)
```bash
curl -X POST http://localhost:8000/api/audit/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Patient: John Doe, Age: 68. Prescription: Warfarin 5mg daily, Aspirin 100mg daily. Symptoms: Chest pain, shortness of breath."
  }'
```

#### Run Audit (File Upload)
```bash
curl -X POST http://localhost:8000/api/audit \
  -F "file=@medical_document.pdf"
```

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/ingest` - Extract text from document
- `POST /api/audit` - Run complete audit (file upload)
- `POST /api/audit/text` - Run complete audit (text input)
- `GET /api/audit/{audit_id}` - Retrieve audit by ID
- `GET /api/analytics/*` - Analytics endpoints (KPIs, trends, symptoms, etc.)

See `http://localhost:8000/docs` for interactive API documentation.

## System Architecture

MedInsight follows a modular architecture with the following key components:

1. **Ingestion Service**: Extracts and processes text from PDFs and images using OCR
2. **NER Service**: Identifies medical entities (drugs, symptoms, vitals, lab values) using BERT-based models
3. **Normalizer Service**: Standardizes and normalizes extracted entities for consistent processing
4. **RAG Service**: Retrieves relevant medical knowledge from the vector store using semantic search
5. **Multi-Agent System**: Five specialized agents perform focused analysis:
   - Dosage Checker: Validates medication dosages
   - Interaction Checker: Identifies drug-drug interactions
   - Red Flag Checker: Detects emergency symptoms
   - Missing Tests Checker: Suggests required diagnostic tests
   - Guideline Compliance Checker: Validates against clinical guidelines
6. **Decision Engine**: Aggregates agent outputs, calculates risk scores, and generates comprehensive reports
7. **Analytics Engine**: Aggregates audit data and provides insights through the dashboard

## Testing

Run the test suite:
```bash
cd backend
source venv/bin/activate
pytest tests/
```

## Configuration

### Environment Variables

Create a `.env` file in the `backend/` directory:

```env
MONGODB_URI=mongodb://localhost:27017/
CHROMA_PERSIST_DIR=./chroma_db
OPENAI_API_KEY=your_openai_api_key_here
```

### Knowledge Base

Add medical knowledge sources to `knowledge_base/sources/`:
- `drug_guidelines.txt` - Drug dosage guidelines
- `interactions.txt` - Drug interaction information
- `red_flags.txt` - Emergency symptom definitions
- `lab_ranges.txt` - Laboratory reference ranges
- `treatment_guidelines.txt` - Clinical treatment guidelines

After adding files, rebuild the vector store:
```bash
cd knowledge_base
python build_vectorstore.py
```

## Development

### Project Structure Details

- **Backend Services**: Located in `backend/app/services/`
  - `ingestion.py` - Document processing
  - `ner.py` - Named entity recognition
  - `normalizer.py` - Entity normalization
  - `rag.py` - RAG pipeline with vector search
  - `agents.py` - Multi-agent system
  - `decision_engine.py` - Risk scoring and report generation
  - `llm_service.py` - LLM integration with fallback

- **API Routes**: Located in `backend/app/routes/`
  - `ingest.py` - Document ingestion endpoints
  - `audit.py` - Audit endpoints
  - `analytics.py` - Analytics endpoints
  - `health.py` - Health check

- **Dashboard**: Located in `dashboard/`
  - `streamlit_app.py` - Main dashboard application
  - `components/` - Reusable UI components
  - `pages/` - Dashboard pages

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## Support

For questions, issues, or feature requests, please open an issue in the repository.
