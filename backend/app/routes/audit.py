from fastapi import APIRouter, HTTPException, UploadFile, File, Body, Request
from app.models.schemas import AuditReport, AuditRequest
from typing import Optional, Union
import json
from app.services.ingestion import IngestionService
from app.services.ner import NERService
from app.services.normalizer import NormalizerService
from app.services.rag import RAGService
from app.services.agents import (
    DosageCheckerAgent,
    InteractionCheckerAgent,
    RedFlagCheckerAgent,
    MissingTestsCheckerAgent,
    GuidelineComplianceCheckerAgent
)
from app.services.decision_engine import DecisionEngine
from app.models.db import MongoDB
import tempfile
import os

router = APIRouter()
ingestion_service = IngestionService()
ner_service = NERService()
normalizer_service = NormalizerService()
rag_service = RAGService()


@router.post("/audit", response_model=AuditReport)
async def run_audit(
    file: UploadFile = File(...)
):
    """Run complete audit pipeline on uploaded file.
    
    **File Upload** (multipart/form-data):
    - Upload PDF or image file
    - File will be processed and audited
    """
    try:
        input_text = None
        
        # Process uploaded file
        file_ext = os.path.splitext(file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            if file.content_type == "application/pdf" or file_ext == ".pdf":
                input_text, _ = ingestion_service.extract_from_pdf(tmp_path)
            elif file.content_type.startswith("image/"):
                input_text, _ = ingestion_service.extract_from_image(tmp_path)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")
            input_text = ingestion_service.cleanup_text(input_text)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        if not input_text:
            raise HTTPException(status_code=400, detail="No text extracted from file")
        
        # Continue with audit pipeline...
        return await _run_audit_pipeline(input_text)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running audit: {str(e)}")


@router.post("/audit/text", response_model=AuditReport)
async def run_audit_text(
    audit_request: AuditRequest
):
    """Run complete audit pipeline on text input.
    
    **JSON Text** (application/json):
    - Send: `{"text": "Patient: John. Prescription: Aspirin 100mg daily."}`
    - OR: `{"file_path": "/path/to/file.pdf"}`
    """
    try:
        input_text = None
        
        if audit_request.text:
            input_text = audit_request.text
        elif audit_request.file_path:
            file_path = audit_request.file_path
            if os.path.exists(file_path):
                if file_path.endswith('.pdf'):
                    input_text, _ = ingestion_service.extract_from_pdf(file_path)
                else:
                    input_text, _ = ingestion_service.extract_from_image(file_path)
                input_text = ingestion_service.cleanup_text(input_text)
            else:
                raise HTTPException(status_code=404, detail="File not found")
        else:
            raise HTTPException(status_code=400, detail="Either 'text' or 'file_path' must be provided")
        
        if not input_text:
            raise HTTPException(status_code=400, detail="No text provided")
        
        # Continue with audit pipeline...
        return await _run_audit_pipeline(input_text)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running audit: {str(e)}")


async def _run_audit_pipeline(input_text: str) -> AuditReport:
    """Internal function to run the audit pipeline."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting audit pipeline for text: {input_text[:100]}...")
    
    # Step 1: NER
    logger.debug("Step 1: Extracting entities with NER")
    ner_result = ner_service.extract_entities(input_text)
    logger.info(f"NER extracted {len(ner_result.entities)} entities")
    
    # Step 2: Normalize entities
    logger.debug("Step 2: Normalizing entities")
    normalized = normalizer_service.normalize_entities(
        ner_result.entities,
        input_text
    )
    ner_result.normalized_entities = normalized
    logger.debug(f"Normalized: {len(normalized.get('drugs', []))} drugs, "
                 f"{len(normalized.get('symptoms', []))} symptoms, "
                 f"{len(normalized.get('vitals', []))} vitals")
    
    # Step 3: Run agents
    logger.debug("Step 3: Running agents")
    agents = [
        DosageCheckerAgent(rag_service),
        InteractionCheckerAgent(rag_service),
        RedFlagCheckerAgent(rag_service),
        MissingTestsCheckerAgent(rag_service),
        GuidelineComplianceCheckerAgent(rag_service)
    ]
    
    agent_outputs = []
    
    # Dosage checker
    logger.debug("Running DosageChecker")
    dosage_result = agents[0].check(
        normalized.get('drugs', []),
        normalized.get('dosages', []),
        normalized.get('frequencies', [])
    )
    agent_outputs.append(dosage_result)
    logger.info(f"DosageChecker: {dosage_result.status.value if hasattr(dosage_result.status, 'value') else dosage_result.status}, score={dosage_result.score}")
    
    # Interaction checker
    logger.debug("Running InteractionChecker")
    interaction_result = agents[1].check(normalized.get('drugs', []))
    agent_outputs.append(interaction_result)
    logger.info(f"InteractionChecker: {interaction_result.status.value if hasattr(interaction_result.status, 'value') else interaction_result.status}, score={interaction_result.score}")
    
    # Red flag checker
    logger.debug("Running RedFlagChecker")
    redflag_result = agents[2].check(
        normalized.get('symptoms', []),
        input_text,
        normalized.get('lab_values', [])
    )
    agent_outputs.append(redflag_result)
    logger.info(f"RedFlagChecker: {redflag_result.status.value if hasattr(redflag_result.status, 'value') else redflag_result.status}, score={redflag_result.score}")
    
    # Missing tests checker
    logger.debug("Running MissingTestsChecker")
    missingtests_result = agents[3].check(
        normalized.get('drugs', []),
        normalized.get('lab_values', [])
    )
    agent_outputs.append(missingtests_result)
    logger.info(f"MissingTestsChecker: {missingtests_result.status.value if hasattr(missingtests_result.status, 'value') else missingtests_result.status}, score={missingtests_result.score}")
    
    # Guideline compliance checker
    logger.debug("Running GuidelineComplianceChecker")
    guideline_result = agents[4].check(
        normalized.get('drugs', []),
        normalized.get('symptoms', []),
        input_text
    )
    agent_outputs.append(guideline_result)
    logger.info(f"GuidelineComplianceChecker: {guideline_result.status.value if hasattr(guideline_result.status, 'value') else guideline_result.status}, score={guideline_result.score}")
    
    # Step 4: Advanced Pattern Detector
    logger.debug("Step 4: Running pattern detector")
    from app.services.pattern_detector import AdvancedPatternDetector
    pattern_detector = AdvancedPatternDetector()
    
    # Extract raw symptoms, vitals, and lab values for the detector
    raw_symptoms = [e.text for e in ner_result.entities if e.type == 'SYMPTOM']
    raw_vitals = [e.text for e in ner_result.entities if e.type == 'VITALS']
    raw_lab_values = [e.text for e in ner_result.entities if e.type == 'LAB_VALUE']
    
    logger.debug(f"Pattern detector inputs: {len(raw_symptoms)} symptoms, {len(raw_vitals)} vitals, {len(raw_lab_values)} lab values")
    
    # Use detect_emergencies method
    pattern_detections = pattern_detector.detect_emergencies(
        raw_symptoms,
        input_text,
        raw_lab_values,
        raw_vitals
    )
    logger.info(f"Pattern detector found {len(pattern_detections)} emergency patterns")
    
    # Step 5: Decision engine (with pattern detections)
    logger.debug("Step 5: Merging agent outputs with DecisionEngine")
    audit_report = DecisionEngine.merge_agent_outputs(
        agent_outputs, 
        ner_result,
        pattern_detections=pattern_detections
    )
    logger.info(f"Final audit report: risk_score={audit_report.risk_score:.2f}, "
               f"critical={len(audit_report.critical_issues)}, "
               f"high={len(audit_report.high_issues)}, "
               f"moderate={len(audit_report.moderate_issues)}")
    
    # Step 6: Save to MongoDB (required)
    logger.debug("Step 6: Saving to MongoDB")
    audit_dict = audit_report.dict()
    audit_id = MongoDB.save_audit(audit_dict)
    audit_report.audit_id = audit_id
    logger.info(f"Audit saved with ID: {audit_id}")
    
    return audit_report


@router.get("/audit/{audit_id}", response_model=AuditReport)
async def get_audit(audit_id: str):
    """Retrieve a past audit by ID."""
    audit_data = MongoDB.get_audit(audit_id)
    if not audit_data:
        raise HTTPException(status_code=404, detail="Audit not found")
    
    # Convert MongoDB document to AuditReport
    audit_data.pop('_id', None)
    audit_data.pop('created_at', None)
    return AuditReport(**audit_data)

