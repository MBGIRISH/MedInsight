"""
Evaluation tests for synthetic medical emergency cases.
Tests accuracy and tracks failures after each update.
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
from app.services.pattern_detector import AdvancedPatternDetector
from app.services.decision_engine import DecisionEngine
from app.models.schemas import Severity


# Synthetic test cases
SYNTHETIC_CASES = {
    "dka_case": {
        "description": "Diabetic Ketoacidosis (DKA) emergency",
        "text": """Patient: John Doe, Age: 28, Type 1 Diabetes
Chief Complaint: Nausea, vomiting, excessive thirst, blurred vision
Symptoms: Polyuria, polydipsia, excessive thirst, blurred vision, nausea, vomiting, dehydration, dry mouth
Vitals: BP 110/70 mmHg, Heart rate 120 bpm, Temperature 99.5°F, Respiratory rate 24/min
Labs: Glucose 450 mg/dL, Ketones positive, pH 7.15
Prescription: Insulin lispro 10 units before meals"""
    },
    "meningitis_case": {
        "description": "Meningitis emergency",
        "text": """Patient: Sarah Smith, Age: 25
Chief Complaint: Severe headache, neck stiffness, photophobia
Symptoms: High fever 103.5°F, severe headache, neck stiffness, nuchal rigidity, photophobia, light sensitivity, vomiting, nausea, confusion
Vitals: BP 130/85 mmHg, Heart rate 110 bpm, Temperature 103.5°F
Labs: WBC 15,000, Glucose 95 mg/dL
Prescription: Ceftriaxone 2g IV, Dexamethasone 10mg IV"""
    },
    "heart_attack_case": {
        "description": "Myocardial Infarction (Heart Attack)",
        "text": """Patient: Robert Johnson, Age: 58
Chief Complaint: Severe chest pain radiating to left arm
Symptoms: Crushing chest pain, chest pain radiating to left arm, profuse sweating, diaphoresis, shortness of breath, dyspnea, nausea, dizziness
Vitals: BP 150/95 mmHg, Heart rate 105 bpm, Oxygen saturation 94%
Labs: Troponin elevated, CK-MB elevated
Prescription: Aspirin 325mg, Clopidogrel 600mg, Atorvastatin 80mg"""
    },
    "stroke_case": {
        "description": "Stroke (CVA) emergency",
        "text": """Patient: Mary Williams, Age: 72
Chief Complaint: Sudden facial droop, arm weakness, speech difficulty
Symptoms: Facial droop, left arm weakness, speech difficulty, slurred speech, confusion, altered mental status, dizziness
Vitals: BP 180/100 mmHg, Heart rate 88 bpm, Temperature 98.6°F
Labs: Glucose 110 mg/dL, INR 1.2
Prescription: Aspirin 325mg, Atorvastatin 40mg
Neurological Exam: Left hemiparesis, aphasia"""
    },
    "sepsis_case": {
        "description": "Sepsis emergency",
        "text": """Patient: David Brown, Age: 65
Chief Complaint: High fever, chills, confusion
Symptoms: High fever 102.8°F, chills, rigors, shaking chills, confusion, altered mental status, rapid breathing, tachypnea, tachycardia
Vitals: BP 85/50 mmHg, Heart rate 130 bpm, Temperature 102.8°F, Respiratory rate 28/min, Oxygen saturation 89%
Labs: WBC 22,000, Lactate 4.5 mmol/L, Glucose 140 mg/dL
Prescription: Vancomycin 1g IV, Piperacillin-tazobactam 4.5g IV"""
    },
    "hypertensive_emergency_case": {
        "description": "Hypertensive crisis emergency",
        "text": """Patient: Linda Davis, Age: 55
Chief Complaint: Severe headache, blurred vision
Symptoms: Severe headache, blurred vision, vision changes, chest pain, shortness of breath, nausea
Vitals: BP 210/130 mmHg, Heart rate 95 bpm, Temperature 98.4°F
Labs: Creatinine 1.8 mg/dL, Glucose 105 mg/dL
Prescription: Labetalol 20mg IV, Amlodipine 10mg daily"""
    }
}


class TestResults:
    """Track test results and accuracy."""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.failures = []
    
    def record_test(self, case_name: str, expected_critical: bool, actual_critical: bool, 
                   risk_score: float, details: dict):
        """Record test result."""
        self.total += 1
        passed = (expected_critical and actual_critical) or (not expected_critical and not actual_critical)
        
        if passed:
            self.passed += 1
        else:
            self.failed += 1
            self.failures.append({
                'case': case_name,
                'expected_critical': expected_critical,
                'actual_critical': actual_critical,
                'risk_score': risk_score,
                'details': details
            })
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("📊 EVALUATION TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {self.total}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"Accuracy: {(self.passed/self.total*100):.1f}%")
        
        if self.failures:
            print("\n❌ FAILURES:")
            for failure in self.failures:
                print(f"  - {failure['case']}: Expected critical={failure['expected_critical']}, "
                      f"Got critical={failure['actual_critical']}, Score={failure['risk_score']:.2f}")
        print("="*60 + "\n")


def run_evaluation_tests():
    """Run all evaluation tests."""
    results = TestResults()
    
    # Initialize services
    ner_service = NERService()
    normalizer_service = NormalizerService()
    rag_service = RAGService()
    pattern_detector = AdvancedPatternDetector()
    
    agents = [
        DosageCheckerAgent(rag_service),
        InteractionCheckerAgent(rag_service),
        RedFlagCheckerAgent(rag_service),
        MissingTestsCheckerAgent(rag_service),
        GuidelineComplianceCheckerAgent(rag_service)
    ]
    
    # Test each case
    for case_name, case_data in SYNTHETIC_CASES.items():
        print(f"\n🔍 Testing: {case_data['description']}")
        text = case_data['text']
        
        # Run pipeline
        ner_result = ner_service.extract_entities(text)
        normalized = normalizer_service.normalize_entities(ner_result.entities, text)
        ner_result.normalized_entities = normalized
        
        # Run agents
        agent_outputs = []
        agent_outputs.append(agents[0].check(
            normalized.get('drugs', []),
            normalized.get('dosages', []),
            normalized.get('frequencies', [])
        ))
        agent_outputs.append(agents[1].check(normalized.get('drugs', [])))
        agent_outputs.append(agents[2].check(
            normalized.get('symptoms', []),
            text,
            normalized.get('lab_values', [])
        ))
        agent_outputs.append(agents[3].check(
            normalized.get('drugs', []),
            normalized.get('lab_values', [])
        ))
        agent_outputs.append(agents[4].check(
            normalized.get('drugs', []),
            normalized.get('symptoms', []),
            text
        ))
        
        # Pattern detector - convert lab_values to list of strings
        raw_symptoms = [e.text for e in ner_result.entities if e.type == 'SYMPTOM']
        raw_vitals = [e.text for e in ner_result.entities if e.type == 'VITALS']
        raw_lab_values = [e.text for e in ner_result.entities if e.type == 'LAB_VALUE']
        
        # Pattern detector expects list of strings for lab_values
        pattern_detections = pattern_detector.detect_emergencies(
            raw_symptoms,
            text,
            raw_lab_values,  # Already list of strings
            raw_vitals
        ) if hasattr(pattern_detector, 'detect_emergencies') else []
        
        # Decision engine
        audit_report = DecisionEngine.merge_agent_outputs(
            agent_outputs,
            ner_result,
            pattern_detections=pattern_detections
        )
        
        # Evaluate
        risk_score = audit_report.risk_score
        critical_issues = audit_report.critical_issues
        is_critical = len(critical_issues) > 0 or risk_score >= 9.0
        
        # All emergency cases should be critical
        expected_critical = True
        
        results.record_test(
            case_name,
            expected_critical,
            is_critical,
            risk_score,
            {
                'critical_issues': len(critical_issues),
                'pattern_detections': len(pattern_detections),
                'risk_score': risk_score
            }
        )
        
        print(f"  Risk Score: {risk_score:.2f}/10")
        print(f"  Critical Issues: {len(critical_issues)}")
        print(f"  Pattern Detections: {len(pattern_detections)}")
        if pattern_detections:
            for det in pattern_detections:
                print(f"    - {det.get('emergency')}: score={det.get('score')}")
        print(f"  Status: {'✅ CRITICAL' if is_critical else '❌ NOT CRITICAL'}")
    
    # Print summary
    results.print_summary()
    
    return results


if __name__ == "__main__":
    results = run_evaluation_tests()
    # Exit with error code if failures
    sys.exit(1 if results.failed > 0 else 0)

