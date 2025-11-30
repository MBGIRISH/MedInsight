from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import IngestResponse, NERResult
from app.services.ingestion import IngestionService
from app.services.ner import NERService
import tempfile
import os

router = APIRouter()
ingestion_service = IngestionService()
ner_service = NERService()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """Ingest PDF or image and extract text with NER."""
    try:
        # Save uploaded file temporarily
        file_ext = os.path.splitext(file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Extract text based on file type
            if file.content_type == "application/pdf" or file_ext == ".pdf":
                text, chunks = ingestion_service.extract_from_pdf(tmp_path)
            elif file.content_type.startswith("image/") or file_ext in [".jpg", ".jpeg", ".png", ".tiff"]:
                text, chunks = ingestion_service.extract_from_image(tmp_path)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file.content_type}"
                )
            
            # Clean text
            text = ingestion_service.cleanup_text(text)
            
            # Extract entities
            ner_result = ner_service.extract_entities(text)
            
            return IngestResponse(
                text=text,
                chunks=chunks,
                ner_result=ner_result
            )
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

