from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    OK = "ok"


# Next Steps Schema
class OrderedItem(BaseModel):
    """Represents an ordered test or procedure."""
    type: str = Field(..., description="Type: lab, imaging, ecg, procedure")
    name: str = Field(..., description="Name of the test/procedure")
    urgency: str = Field(..., description="stat, 24h, 72h, routine")
    notes: Optional[str] = Field(None, description="Additional notes")


class TreatmentRecommendation(BaseModel):
    """Represents a treatment recommendation."""
    drug: str = Field(..., description="Drug name")
    dose: str = Field(..., description="Dose and route (e.g., '500-1000 mg PO q4-6h PRN')")
    max_per_day: Optional[str] = Field(None, description="Maximum daily dose")
    notes: Optional[str] = Field(None, description="Additional notes or conditions")
    contraindications: Optional[List[str]] = Field(None, description="List of contraindications")
    human_approval_required: bool = Field(False, description="Whether human approval is required")


class MonitoringParameter(BaseModel):
    """Represents a monitoring parameter."""
    parameter: str = Field(..., description="What to monitor (e.g., 'Oxygen saturation')")
    target: str = Field(..., description="Target value or range (e.g., '>=94%')")
    frequency: str = Field(..., description="How often to monitor (e.g., 'q4h')")
    method: Optional[str] = Field(None, description="Method of monitoring (e.g., 'pulse oximetry')")


class NextStepItem(BaseModel):
    """A single actionable next step item."""
    title: str = Field(..., description="Short title for the action")
    priority: str = Field(..., description="urgent, high, medium, low")
    action_type: str = Field(..., description="Order Test | Start Treatment | Monitor | Refer | Admit | Provide Discharge Advice")
    recommended_by_agent: str = Field(..., description="Agent that recommended this")
    rationale: str = Field(..., description="Why this is needed (1-2 sentences)")
    ordered_items: Optional[List[OrderedItem]] = Field(None, description="Tests/procedures to order")
    treatment_recommendations: Optional[List[TreatmentRecommendation]] = Field(None, description="Treatment recommendations")
    monitoring_parameters: Optional[List[MonitoringParameter]] = Field(None, description="What to monitor")
    disposition: str = Field(..., description="ER | Admit | OPD | Home with follow-up")
    clinical_confidence: str = Field(..., description="high, medium, low")
    evidence_ids: Optional[List[str]] = Field(None, description="IDs of retrieved knowledge chunks")


class NextSteps(BaseModel):
    """Structured next steps for the audit report."""
    summary: str = Field(..., description="One-line summary of actions")
    urgency_level: str = Field(..., description="immediate, 24h, 72h, routine")
    items: List[NextStepItem] = Field(default_factory=list, description="List of actionable items")
    patient_instructions: str = Field(..., description="Simple, 2-3 line instructions for the patient")
    clinician_note: str = Field(..., description="Short note for clinician")
    disclaimer: str = Field(
        default="Automated triage aid — confirm with clinician. Follow local protocols. This tool does not replace clinical judgement.",
        description="Safety disclaimer"
    )


class EntityType(str, Enum):
    DRUG = "DRUG"
    DOSAGE = "DOSAGE"
    FREQUENCY = "FREQUENCY"
    DURATION = "DURATION"
    SYMPTOM = "SYMPTOM"
    LAB_VALUE = "LAB_VALUE"
    VITALS = "VITALS"


class ExtractedEntity(BaseModel):
    text: str
    type: EntityType
    start: int
    end: int
    confidence: float = 0.0


class NERResult(BaseModel):
    entities: List[ExtractedEntity]
    raw_text: str
    normalized_entities: Dict[str, Any] = {}


class AgentOutput(BaseModel):
    agent: str
    status: Severity
    message: str
    evidence: List[str] = []
    score: float = 0.0
    details: Dict[str, Any] = {}


class AuditReport(BaseModel):
    audit_id: str
    timestamp: datetime
    critical_issues: List[Dict[str, Any]] = []
    high_issues: List[Dict[str, Any]] = []
    moderate_issues: List[Dict[str, Any]] = []
    safe_items: List[Dict[str, Any]] = []
    recommendations: List[str] = []
    risk_score: float = Field(ge=0, le=10)
    agent_outputs: List[AgentOutput] = []
    ner_result: Optional[NERResult] = None
    next_steps: Optional[NextSteps] = Field(None, description="Structured actionable next steps")


class IngestResponse(BaseModel):
    text: str
    chunks: List[str]
    ner_result: Optional[NERResult] = None


class AuditRequest(BaseModel):
    text: Optional[str] = None
    file_path: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, str] = {}

