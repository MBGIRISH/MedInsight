"""
Next Steps Generator for Medical Audit Reports.
Produces clinician-grade, prioritized, actionable recommendations.
"""
from typing import List, Dict, Any, Optional
from app.models.schemas import (
    NextSteps, NextStepItem, OrderedItem, TreatmentRecommendation,
    MonitoringParameter, Severity, AgentOutput
)
import logging

# Forward reference handling
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.models.schemas import NextSteps

logger = logging.getLogger(__name__)


class NextStepsGenerator:
    """Generates structured, actionable next steps from audit results."""
    
    def __init__(self):
        self.critical_meds_requiring_approval = [
            'n-acetylcysteine', 'nac', 'insulin', 'epinephrine',
            'dopamine', 'norepinephrine', 'antibiotics iv', 'anticoagulants'
        ]
    
    def generate_next_steps(
        self,
        critical_issues: List[Dict[str, Any]],
        high_issues: List[Dict[str, Any]],
        moderate_issues: List[Dict[str, Any]],
        low_issues: List[Dict[str, Any]],
        agent_outputs: List[AgentOutput],
        ner_result: Optional[Any] = None,
        risk_score: float = 0.0
    ) -> NextSteps:
        """Generate structured next steps based on audit findings."""
        logger.info("Generating next steps from audit results")
        
        # Determine urgency level
        urgency_level = self._determine_urgency_level(
            critical_issues, high_issues, moderate_issues, low_issues, risk_score
        )
        
        # Generate items based on severity
        items = []
        
        if critical_issues:
            items.extend(self._generate_critical_items(critical_issues, agent_outputs, ner_result))
        
        if high_issues:
            items.extend(self._generate_high_items(high_issues, agent_outputs, ner_result))
        
        if moderate_issues:
            items.extend(self._generate_moderate_items(moderate_issues, agent_outputs, ner_result))
        
        if low_issues:
            items.extend(self._generate_low_items(low_issues, agent_outputs, ner_result))
        
        # If no issues, generate routine follow-up
        if not items:
            items = self._generate_routine_items(ner_result)
        
        # Generate summary
        summary = self._generate_summary(urgency_level, items, risk_score)
        
        # Generate patient instructions
        patient_instructions = self._generate_patient_instructions(
            urgency_level, items, critical_issues, high_issues
        )
        
        # Generate clinician note
        clinician_note = self._generate_clinician_note(
            urgency_level, items, risk_score, agent_outputs
        )
        
        return NextSteps(
            summary=summary,
            urgency_level=urgency_level,
            items=items,
            patient_instructions=patient_instructions,
            clinician_note=clinician_note
        )
    
    def _determine_urgency_level(
        self,
        critical_issues: List[Dict],
        high_issues: List[Dict],
        moderate_issues: List[Dict],
        low_issues: List[Dict],
        risk_score: float
    ) -> str:
        """Determine overall urgency level."""
        if critical_issues or risk_score >= 9.0:
            return "immediate"
        elif high_issues or risk_score >= 6.0:
            return "24h"
        elif moderate_issues or risk_score >= 3.0:
            return "72h"
        else:
            return "routine"
    
    def _generate_critical_items(
        self,
        critical_issues: List[Dict[str, Any]],
        agent_outputs: List[AgentOutput],
        ner_result: Optional[Any]
    ) -> List[NextStepItem]:
        """Generate items for critical issues."""
        items = []
        
        for issue in critical_issues:
            agent = issue.get('agent', '')
            message = issue.get('message', '')
            evidence = issue.get('evidence', [])
            evidence_ids = self._extract_evidence_ids(evidence)
            
            if agent == 'RedFlagChecker' or agent == 'PatternDetector':
                # Emergency red flag - generate ER evaluation item
                items.append(NextStepItem(
                    title="Immediate Emergency Evaluation",
                    priority="urgent",
                    action_type="Refer",
                    recommended_by_agent=agent,
                    rationale=f"Critical red flag symptoms detected: {message[:200]}. Life-threatening emergency requiring immediate medical attention.",
                    ordered_items=[
                        OrderedItem(type="lab", name="CBC with differential", urgency="stat", notes="Complete blood count"),
                        OrderedItem(type="lab", name="Basic Metabolic Panel", urgency="stat", notes="Electrolytes, BUN, Creatinine"),
                        OrderedItem(type="lab", name="Lactate", urgency="stat", notes="If sepsis suspected"),
                        OrderedItem(type="ecg", name="12-lead ECG", urgency="stat", notes="If cardiac symptoms"),
                    ],
                    treatment_recommendations=None,  # ER will determine treatment
                    monitoring_parameters=[
                        MonitoringParameter(parameter="Vital signs", target="Per protocol", frequency="q15min", method="Continuous monitoring"),
                        MonitoringParameter(parameter="Oxygen saturation", target=">=94%", frequency="continuous", method="Pulse oximetry"),
                    ],
                    disposition="ER",
                    clinical_confidence="high",
                    evidence_ids=evidence_ids
                ))
            
            elif agent == 'DosageChecker':
                # Critical dosage issue
                items.append(NextStepItem(
                    title="Immediate Dose Adjustment Required",
                    priority="urgent",
                    action_type="Start Treatment",
                    recommended_by_agent=agent,
                    rationale=f"Critical medication dosing issue: {message[:200]}. Immediate dose correction required to prevent adverse effects.",
                    ordered_items=[
                        OrderedItem(type="lab", name="Drug level (if applicable)", urgency="stat", notes="For drugs with narrow therapeutic index"),
                        OrderedItem(type="lab", name="LFTs", urgency="stat", notes="If hepatotoxic drug"),
                        OrderedItem(type="lab", name="Renal function", urgency="stat", notes="If renally cleared drug"),
                    ],
                    treatment_recommendations=None,  # Requires clinician review
                    monitoring_parameters=[
                        MonitoringParameter(parameter="Clinical response", target="Improvement", frequency="q4h", method="Clinical assessment"),
                        MonitoringParameter(parameter="Adverse effects", target="None", frequency="q4h", method="Clinical assessment"),
                    ],
                    disposition="ER or Admit",
                    clinical_confidence="high",
                    evidence_ids=evidence_ids
                ))
            
            elif agent == 'InteractionChecker':
                # Critical drug interaction
                items.append(NextStepItem(
                    title="Discontinue Interacting Medication",
                    priority="urgent",
                    action_type="Start Treatment",
                    recommended_by_agent=agent,
                    rationale=f"Critical drug-drug interaction detected: {message[:200]}. One medication must be discontinued or switched immediately.",
                    ordered_items=[
                        OrderedItem(type="lab", name="Drug levels (if applicable)", urgency="stat", notes=""),
                        OrderedItem(type="lab", name="Coagulation studies", urgency="stat", notes="If anticoagulant interaction"),
                    ],
                    treatment_recommendations=None,  # Requires clinician decision
                    monitoring_parameters=[
                        MonitoringParameter(parameter="Signs of interaction", target="None", frequency="q4h", method="Clinical assessment"),
                    ],
                    disposition="ER or Admit",
                    clinical_confidence="high",
                    evidence_ids=evidence_ids
                ))
        
        return items
    
    def _generate_high_items(
        self,
        high_issues: List[Dict[str, Any]],
        agent_outputs: List[AgentOutput],
        ner_result: Optional[Any]
    ) -> List[NextStepItem]:
        """Generate items for high priority issues."""
        items = []
        
        # Check for special patterns first
        pneumonia_detected = self._detect_pneumonia_pattern(high_issues, ner_result)
        overdose_detected = self._detect_overdose_pattern(high_issues, agent_outputs, ner_result)
        
        if overdose_detected:
            items.extend(self._generate_overdose_items(high_issues, agent_outputs, ner_result))
        elif pneumonia_detected:
            items.extend(self._generate_pneumonia_items(high_issues, agent_outputs, ner_result))
        else:
            # Generic high priority items
            for issue in high_issues:
                agent = issue.get('agent', '')
                message = issue.get('message', '')
                evidence = issue.get('evidence', [])
                evidence_ids = self._extract_evidence_ids(evidence)
                
                if agent == 'RedFlagChecker':
                    items.append(NextStepItem(
                        title="Urgent Clinical Evaluation",
                        priority="high",
                        action_type="Order Test",
                        recommended_by_agent=agent,
                        rationale=f"High-priority clinical findings: {message[:200]}. Urgent evaluation recommended within 24 hours.",
                        ordered_items=[
                            OrderedItem(type="lab", name="CBC", urgency="24h", notes="Complete blood count"),
                            OrderedItem(type="lab", name="CRP or ESR", urgency="24h", notes="Inflammatory markers"),
                        ],
                        treatment_recommendations=None,
                        monitoring_parameters=[
                            MonitoringParameter(parameter="Symptoms", target="Improvement", frequency="q8h", method="Patient self-report"),
                        ],
                        disposition="OPD with urgent follow-up",
                        clinical_confidence="medium",
                        evidence_ids=evidence_ids
                    ))
                
                elif agent == 'MissingTestsChecker':
                    items.append(NextStepItem(
                        title="Order Essential Laboratory Tests",
                        priority="high",
                        action_type="Order Test",
                        recommended_by_agent=agent,
                        rationale=f"Essential monitoring tests required: {message[:200]}. Baseline labs needed before continuing medication.",
                        ordered_items=self._generate_required_tests(issue, evidence_ids),
                        treatment_recommendations=None,
                        monitoring_parameters=None,
                        disposition="OPD",
                        clinical_confidence="high",
                        evidence_ids=evidence_ids
                    ))
        
        return items
    
    def _detect_pneumonia_pattern(
        self,
        high_issues: List[Dict[str, Any]],
        ner_result: Optional[Any]
    ) -> bool:
        """Detect if this is a pneumonia-like case."""
        if not ner_result or not hasattr(ner_result, 'normalized_entities'):
            return False
        
        normalized = ner_result.normalized_entities
        symptoms = normalized.get('symptoms', [])
        vitals = normalized.get('vitals', [])
        raw_text = getattr(ner_result, 'raw_text', '') if hasattr(ner_result, 'raw_text') else ''
        
        # Check for pneumonia indicators in symptoms
        has_fever = any('fever' in str(s).lower() or '101' in str(s).lower() or 'temperature' in str(s).lower() for s in symptoms) or 'fever' in raw_text.lower() or '101' in raw_text.lower()
        has_cough = any('cough' in str(s).lower() or 'sputum' in str(s).lower() for s in symptoms) or 'cough' in raw_text.lower() or 'sputum' in raw_text.lower()
        has_sob = any('shortness of breath' in str(s).lower() or 'dyspnea' in str(s).lower() or 'breath' in str(s).lower() for s in symptoms) or 'shortness of breath' in raw_text.lower() or 'dyspnea' in raw_text.lower()
        
        # Also check vitals for elevated HR (100-120 range)
        has_elevated_hr = False
        for vital in vitals:
            if isinstance(vital, dict):
                if 'heart' in str(vital.get('type', '')).lower() or 'hr' in str(vital.get('type', '')).lower():
                    hr_value = vital.get('value')
                    if hr_value and 100 <= hr_value <= 120:
                        has_elevated_hr = True
                        break
        
        # Pneumonia pattern: fever + (cough OR sputum) + (SOB OR elevated HR)
        return has_fever and (has_cough or has_sob) and (has_sob or has_elevated_hr)
    
    def _generate_pneumonia_items(
        self,
        high_issues: List[Dict[str, Any]],
        agent_outputs: List[AgentOutput],
        ner_result: Optional[Any]
    ) -> List[NextStepItem]:
        """Generate specific items for pneumonia case."""
        items = []
        
        # Extract evidence IDs
        evidence_ids = []
        for issue in high_issues:
            evidence = issue.get('evidence', [])
            evidence_ids.extend(self._extract_evidence_ids(evidence))
        
        # Item 1: Order tests
        items.append(NextStepItem(
            title="Order Diagnostic Tests for Pneumonia",
            priority="high",
            action_type="Order Test",
            recommended_by_agent="RedFlagChecker",
            rationale="Pneumonia pattern detected (fever, productive cough, elevated HR, SOB). Diagnostic imaging and labs needed to confirm diagnosis and guide treatment.",
            ordered_items=[
                OrderedItem(type="imaging", name="Chest X-ray", urgency="24h", notes="PA and lateral views"),
                OrderedItem(type="lab", name="CBC with differential", urgency="24h", notes="White blood cell count"),
                OrderedItem(type="lab", name="CRP", urgency="24h", notes="C-reactive protein"),
                OrderedItem(type="lab", name="Blood cultures", urgency="stat", notes="If febrile >38.5°C or severe symptoms"),
            ],
            treatment_recommendations=None,
            monitoring_parameters=None,
            disposition="OPD",
            clinical_confidence="high",
            evidence_ids=evidence_ids[:3] if evidence_ids else None
        ))
        
        # Item 2: Start empiric antibiotic (with approval flag)
        items.append(NextStepItem(
            title="Consider Empiric Antibiotic Therapy",
            priority="high",
            action_type="Start Treatment",
            recommended_by_agent="GuidelineComplianceChecker",
            rationale="If pneumonia confirmed and no contraindications, start empiric oral antibiotic per local guidelines. Common options include amoxicillin-clavulanate or doxycycline for community-acquired pneumonia.",
            ordered_items=None,
            treatment_recommendations=[
                TreatmentRecommendation(
                    drug="Amoxicillin-clavulanate",
                    dose="625 mg PO TID",
                    max_per_day="1875 mg",
                    notes="Use if no penicillin allergy. Alternative: Doxycycline 100 mg PO BID if penicillin allergic.",
                    contraindications=["Penicillin allergy", "Severe hepatic impairment"],
                    human_approval_required=True
                ),
                TreatmentRecommendation(
                    drug="Doxycycline",
                    dose="100 mg PO BID",
                    max_per_day="200 mg",
                    notes="Alternative if penicillin allergic. Not for children <8 years or pregnancy.",
                    contraindications=["Pregnancy", "Children <8 years", "Severe hepatic impairment"],
                    human_approval_required=True
                ),
            ],
            monitoring_parameters=None,
            disposition="OPD",
            clinical_confidence="medium",
            evidence_ids=evidence_ids[:2] if evidence_ids else None
        ))
        
        # Item 3: Monitor oxygen
        items.append(NextStepItem(
            title="Monitor Oxygen Saturation",
            priority="high",
            action_type="Monitor",
            recommended_by_agent="RedFlagChecker",
            rationale="Respiratory symptoms present. Monitor oxygen saturation to detect hypoxia early. If SpO2 < 94%, provide supplemental oxygen and escalate to ER.",
            ordered_items=None,
            treatment_recommendations=None,
            monitoring_parameters=[
                MonitoringParameter(parameter="Oxygen saturation", target=">=94%", frequency="q4h", method="Pulse oximetry"),
                MonitoringParameter(parameter="Respiratory rate", target="12-20/min", frequency="q4h", method="Clinical assessment"),
            ],
            disposition="Home with follow-up if stable; ER if SpO2<94% or worsening",
            clinical_confidence="high",
            evidence_ids=evidence_ids[:1] if evidence_ids else None
        ))
        
        # Item 4: Symptomatic care
        items.append(NextStepItem(
            title="Symptomatic Treatment",
            priority="medium",
            action_type="Start Treatment",
            recommended_by_agent="GuidelineComplianceChecker",
            rationale="Provide symptomatic relief for fever and discomfort while treating underlying infection.",
            ordered_items=None,
            treatment_recommendations=[
                TreatmentRecommendation(
                    drug="Paracetamol (Acetaminophen)",
                    dose="500-1000 mg PO q4-6h PRN",
                    max_per_day="3000 mg",
                    notes="Use if temperature > 38.5°C or significant discomfort. Do not exceed 3000 mg/day.",
                    contraindications=["Severe hepatic impairment", "Active liver disease"],
                    human_approval_required=False
                ),
            ],
            monitoring_parameters=None,
            disposition="Home",
            clinical_confidence="high",
            evidence_ids=None
        ))
        
        return items
    
    def _detect_overdose_pattern(
        self,
        high_issues: List[Dict[str, Any]],
        agent_outputs: List[AgentOutput],
        ner_result: Optional[Any]
    ) -> bool:
        """Detect if this is an overdose case (e.g., paracetamol overdose)."""
        # Check agent outputs for dosage issues
        for agent_output in agent_outputs:
            if agent_output.agent == 'DosageChecker' and agent_output.status in [Severity.CRITICAL, Severity.HIGH]:
                message = agent_output.message.lower()
                if 'overdose' in message or 'excessive' in message or 'too high' in message:
                    # Check for paracetamol/acetaminophen
                    if ner_result and hasattr(ner_result, 'normalized_entities'):
                        drugs = ner_result.normalized_entities.get('drugs', [])
                        for drug in drugs:
                            drug_str = str(drug).lower()
                            if 'paracetamol' in drug_str or 'acetaminophen' in drug_str or 'tylenol' in drug_str:
                                return True
        return False
    
    def _generate_overdose_items(
        self,
        high_issues: List[Dict[str, Any]],
        agent_outputs: List[AgentOutput],
        ner_result: Optional[Any]
    ) -> List[NextStepItem]:
        """Generate specific items for overdose case (e.g., paracetamol)."""
        items = []
        
        # Extract evidence IDs
        evidence_ids = []
        for issue in high_issues:
            evidence = issue.get('evidence', [])
            evidence_ids.extend(self._extract_evidence_ids(evidence))
        
        # Check if paracetamol overdose
        is_paracetamol = False
        if ner_result and hasattr(ner_result, 'normalized_entities'):
            drugs = ner_result.normalized_entities.get('drugs', [])
            for drug in drugs:
                drug_str = str(drug).lower()
                if 'paracetamol' in drug_str or 'acetaminophen' in drug_str or 'tylenol' in drug_str:
                    is_paracetamol = True
                    break
        
        if is_paracetamol:
            # Item 1: Stop medication immediately
            items.append(NextStepItem(
                title="Stop Paracetamol Immediately",
                priority="urgent",
                action_type="Start Treatment",
                recommended_by_agent="DosageChecker",
                rationale="Paracetamol overdose detected. Immediate cessation required to prevent further hepatotoxicity. Time since ingestion is critical for treatment decisions.",
                ordered_items=None,
                treatment_recommendations=None,
                monitoring_parameters=None,
                disposition="ER",
                clinical_confidence="high",
                evidence_ids=evidence_ids[:2] if evidence_ids else None
            ))
            
            # Item 2: Order stat labs
            items.append(NextStepItem(
                title="Order STAT Laboratory Tests",
                priority="urgent",
                action_type="Order Test",
                recommended_by_agent="DosageChecker",
                rationale="Immediate laboratory evaluation required to assess paracetamol level, liver function, and coagulation status. Acetaminophen level must be checked within 4-24 hours post-ingestion to determine need for N-acetylcysteine (NAC) therapy.",
                ordered_items=[
                    OrderedItem(type="lab", name="Serum acetaminophen level", urgency="stat", notes="Critical - check within 4-24h post-ingestion"),
                    OrderedItem(type="lab", name="LFTs (AST/ALT)", urgency="stat", notes="Liver function tests"),
                    OrderedItem(type="lab", name="INR", urgency="stat", notes="Coagulation status"),
                    OrderedItem(type="lab", name="BUN/Creatinine", urgency="stat", notes="Renal function"),
                ],
                treatment_recommendations=None,
                monitoring_parameters=None,
                disposition="ER",
                clinical_confidence="high",
                evidence_ids=evidence_ids[:3] if evidence_ids else None
            ))
            
            # Item 3: Start NAC if indicated
            items.append(NextStepItem(
                title="Start N-Acetylcysteine (NAC) if Indicated",
                priority="urgent",
                action_type="Start Treatment",
                recommended_by_agent="DosageChecker",
                rationale="If acetaminophen level is above treatment line on Rumack-Matthew nomogram, start N-acetylcysteine immediately. NAC is most effective when started within 8 hours of ingestion but can be beneficial up to 24 hours.",
                ordered_items=None,
                treatment_recommendations=[
                    TreatmentRecommendation(
                        drug="N-Acetylcysteine (NAC)",
                        dose="IV: 150 mg/kg in 200 mL D5W over 60 min, then 50 mg/kg in 500 mL D5W over 4h, then 100 mg/kg in 1000 mL D5W over 16h. OR Oral: 140 mg/kg loading, then 70 mg/kg q4h for 17 doses",
                        max_per_day="As per protocol",
                        notes="Use IV route if patient cannot tolerate oral or has active vomiting. Check acetaminophen level first to determine if treatment needed. Follow local protocol.",
                        contraindications=["Known hypersensitivity to NAC"],
                        human_approval_required=True
                    ),
                ],
                monitoring_parameters=[
                    MonitoringParameter(parameter="Acetaminophen level", target="Below treatment line", frequency="q4h until declining", method="Serum level"),
                    MonitoringParameter(parameter="LFTs (AST/ALT)", target="Normal or improving", frequency="q12h", method="Laboratory"),
                    MonitoringParameter(parameter="INR", target="<1.5", frequency="q12h", method="Laboratory"),
                ],
                disposition="ER/Admit",
                clinical_confidence="high",
                evidence_ids=evidence_ids[:2] if evidence_ids else None
            ))
        
        return items
    
    def _generate_moderate_items(
        self,
        moderate_issues: List[Dict[str, Any]],
        agent_outputs: List[AgentOutput],
        ner_result: Optional[Any]
    ) -> List[NextStepItem]:
        """Generate items for moderate priority issues."""
        items = []
        
        for issue in moderate_issues:
            agent = issue.get('agent', '')
            message = issue.get('message', '')
            evidence = issue.get('evidence', [])
            evidence_ids = self._extract_evidence_ids(evidence)
            
            items.append(NextStepItem(
                title="Routine Clinical Evaluation",
                priority="medium",
                action_type="Order Test",
                recommended_by_agent=agent,
                rationale=f"Moderate concern identified: {message[:200]}. Routine evaluation recommended within 72 hours.",
                ordered_items=[
                    OrderedItem(type="lab", name="Basic labs", urgency="72h", notes="As clinically indicated"),
                ],
                treatment_recommendations=None,
                monitoring_parameters=[
                    MonitoringParameter(parameter="Symptoms", target="Stable or improving", frequency="daily", method="Patient self-report"),
                ],
                disposition="OPD",
                clinical_confidence="medium",
                evidence_ids=evidence_ids
            ))
        
        return items
    
    def _generate_low_items(
        self,
        low_issues: List[Dict[str, Any]],
        agent_outputs: List[AgentOutput],
        ner_result: Optional[Any]
    ) -> List[NextStepItem]:
        """Generate items for low priority issues."""
        items = []
        
        for issue in low_issues:
            agent = issue.get('agent', '')
            message = issue.get('message', '')
            evidence = issue.get('evidence', [])
            evidence_ids = self._extract_evidence_ids(evidence)
            
            items.append(NextStepItem(
                title="Monitor and Follow-up",
                priority="low",
                action_type="Monitor",
                recommended_by_agent=agent,
                rationale=f"Minor finding: {message[:200]}. Continue monitoring and consider evaluation if symptoms persist or worsen.",
                ordered_items=None,
                treatment_recommendations=None,
                monitoring_parameters=[
                    MonitoringParameter(parameter="Symptoms", target="Stable", frequency="as needed", method="Patient self-report"),
                ],
                disposition="Home with routine follow-up",
                clinical_confidence="low",
                evidence_ids=evidence_ids
            ))
        
        return items
    
    def _generate_routine_items(
        self,
        ner_result: Optional[Any]
    ) -> List[NextStepItem]:
        """Generate routine items for healthy patients."""
        return [
            NextStepItem(
                title="Routine Health Maintenance",
                priority="low",
                action_type="Provide Discharge Advice",
                recommended_by_agent="System",
                rationale="No significant findings detected. Continue routine health maintenance and preventive care.",
                ordered_items=None,
                treatment_recommendations=None,
                monitoring_parameters=None,
                disposition="Home",
                clinical_confidence="high",
                evidence_ids=None
            )
        ]
    
    def _generate_required_tests(
        self,
        issue: Dict[str, Any],
        evidence_ids: List[str]
    ) -> List[OrderedItem]:
        """Generate required tests based on missing tests checker issue."""
        message = issue.get('message', '').lower()
        tests = []
        
        if 'warfarin' in message or 'inr' in message:
            tests.append(OrderedItem(type="lab", name="INR", urgency="24h", notes="International normalized ratio"))
            tests.append(OrderedItem(type="lab", name="PT", urgency="24h", notes="Prothrombin time"))
        
        if 'digoxin' in message:
            tests.append(OrderedItem(type="lab", name="Digoxin level", urgency="24h", notes="Therapeutic range 0.8-2.0 ng/mL"))
            tests.append(OrderedItem(type="lab", name="Creatinine", urgency="24h", notes="Renal function"))
            tests.append(OrderedItem(type="lab", name="eGFR", urgency="24h", notes="Estimated glomerular filtration rate"))
        
        if 'lithium' in message:
            tests.append(OrderedItem(type="lab", name="Lithium level", urgency="24h", notes="Therapeutic range 0.6-1.2 mEq/L"))
            tests.append(OrderedItem(type="lab", name="TSH", urgency="24h", notes="Thyroid function"))
            tests.append(OrderedItem(type="lab", name="Creatinine", urgency="24h", notes="Renal function"))
        
        if 'statin' in message or 'atorvastatin' in message or 'simvastatin' in message:
            tests.append(OrderedItem(type="lab", name="LFTs", urgency="24h", notes="Liver function tests: ALT, AST"))
            tests.append(OrderedItem(type="lab", name="CK", urgency="24h", notes="Creatine kinase for rhabdomyolysis"))
        
        if 'metformin' in message:
            tests.append(OrderedItem(type="lab", name="Creatinine", urgency="24h", notes="Renal function"))
            tests.append(OrderedItem(type="lab", name="eGFR", urgency="24h", notes="Estimated glomerular filtration rate"))
            tests.append(OrderedItem(type="lab", name="B12", urgency="routine", notes="Vitamin B12 level"))
        
        if not tests:
            # Default tests
            tests.append(OrderedItem(type="lab", name="Basic metabolic panel", urgency="24h", notes="As clinically indicated"))
        
        return tests
    
    def _extract_evidence_ids(self, evidence: List[Any]) -> List[str]:
        """Extract evidence IDs from evidence list."""
        evidence_ids = []
        for ev in evidence:
            if isinstance(ev, dict) and 'id' in ev:
                evidence_ids.append(ev['id'])
            elif isinstance(ev, str):
                # Try to extract ID from string (format: "id:xxx" or similar)
                import re
                match = re.search(r'id[:\s]+([a-zA-Z0-9_-]+)', ev, re.IGNORECASE)
                if match:
                    evidence_ids.append(match.group(1))
                else:
                    # Use hash of evidence as ID
                    evidence_ids.append(f"ev_{hash(ev) % 10000}")
        return evidence_ids[:5]  # Limit to 5 IDs
    
    def _generate_summary(
        self,
        urgency_level: str,
        items: List[NextStepItem],
        risk_score: float
    ) -> str:
        """Generate one-line summary."""
        if urgency_level == "immediate":
            return f"CRITICAL: Immediate emergency evaluation required (Risk Score: {risk_score:.1f}/10). {len(items)} urgent action(s) needed."
        elif urgency_level == "24h":
            return f"HIGH PRIORITY: Urgent evaluation within 24 hours recommended (Risk Score: {risk_score:.1f}/10). {len(items)} action(s) required."
        elif urgency_level == "72h":
            return f"MODERATE: Clinical evaluation within 72 hours recommended (Risk Score: {risk_score:.1f}/10). {len(items)} action(s) suggested."
        else:
            return f"ROUTINE: No urgent actions required (Risk Score: {risk_score:.1f}/10). Continue routine monitoring."
    
    def _generate_patient_instructions(
        self,
        urgency_level: str,
        items: List[NextStepItem],
        critical_issues: List[Dict],
        high_issues: List[Dict]
    ) -> str:
        """Generate patient-friendly instructions."""
        if urgency_level == "immediate":
            return "⚠️ URGENT: Go to the emergency department immediately. Do not delay. If you have severe chest pain, difficulty breathing, confusion, or severe symptoms, call emergency services (911) right away."
        elif urgency_level == "24h":
            instructions = "Seek medical care within 24 hours. "
            if any('pneumonia' in str(item.title).lower() or 'cough' in str(item.title).lower() for item in items):
                instructions += "Rest, drink plenty of fluids, and take paracetamol (500-1000 mg every 6 hours as needed, maximum 3000 mg per day) if you have fever or discomfort. Return to the emergency department immediately if you develop severe difficulty breathing, chest pain, confusion, or if your symptoms worsen significantly."
            else:
                instructions += "Monitor your symptoms and return to care if they worsen or if you develop new concerning symptoms."
            return instructions
        elif urgency_level == "72h":
            return "Schedule a follow-up appointment within 72 hours. Continue monitoring your symptoms at home. Return to care sooner if symptoms worsen or if you develop new concerning symptoms."
        else:
            return "No urgent action needed. Continue your usual activities. Return for any new or worsening symptoms. Maintain routine health maintenance and preventive care."
    
    def _generate_clinician_note(
        self,
        urgency_level: str,
        items: List[NextStepItem],
        risk_score: float,
        agent_outputs: List[AgentOutput]
    ) -> str:
        """Generate short note for clinician."""
        agent_names = [a.agent for a in agent_outputs if a.status != Severity.OK]
        agents_str = ", ".join(set(agent_names)) if agent_names else "System"
        
        if urgency_level == "immediate":
            return f"CRITICAL FINDINGS: Risk score {risk_score:.1f}/10. Immediate emergency evaluation required. Triggered by: {agents_str}. Review all ordered items and treatment recommendations. Human approval required for all critical medications."
        elif urgency_level == "24h":
            return f"HIGH PRIORITY: Risk score {risk_score:.1f}/10. Urgent evaluation within 24 hours recommended. Triggered by: {agents_str}. Review treatment recommendations - human approval required for antibiotics and other high-risk medications."
        elif urgency_level == "72h":
            return f"MODERATE CONCERNS: Risk score {risk_score:.1f}/10. Routine evaluation within 72 hours recommended. Triggered by: {agents_str}."
        else:
            return f"ROUTINE: Risk score {risk_score:.1f}/10. No urgent actions required. Continue standard monitoring."

