"""
Unit tests for Next Steps Generator.
"""
import pytest
from app.services.next_steps_generator import NextStepsGenerator
from app.models.schemas import (
    Severity, AgentOutput, NERResult, ExtractedEntity, EntityType
)


def test_pneumonia_case():
    """Test next steps generation for pneumonia-like case."""
    generator = NextStepsGenerator()
    
    # Create mock high issues (pneumonia pattern)
    high_issues = [{
        'agent': 'RedFlagChecker',
        'message': 'HIGH PRIORITY: Pneumonia pattern detected',
        'evidence': ['guideline_chunk_1', 'guideline_chunk_2'],
        'score': 7.0
    }]
    
    # Create NER result with pneumonia symptoms
    ner_result = NERResult(
        entities=[],
        raw_text='Patient reports fever of 101.8°F, cough with yellow sputum, fatigue, and mild shortness of breath. Heart rate: 104 bpm.',
        normalized_entities={
            'symptoms': ['fever', 'cough', 'yellow sputum', 'shortness of breath'],
            'vitals': [{'type': 'temperature', 'value': 101.8}, {'type': 'heart_rate', 'value': 104}],
            'drugs': []
        }
    )
    
    agent_outputs = [
        AgentOutput(
            agent='RedFlagChecker',
            status=Severity.HIGH,
            message='Pneumonia pattern detected',
            evidence=['guideline_chunk_1'],
            score=7.0
        )
    ]
    
    next_steps = generator.generate_next_steps(
        critical_issues=[],
        high_issues=high_issues,
        moderate_issues=[],
        low_issues=[],
        agent_outputs=agent_outputs,
        ner_result=ner_result,
        risk_score=7.0
    )
    
    # Assertions
    assert next_steps.urgency_level == "24h"
    assert len(next_steps.items) >= 3  # Should have multiple items for pneumonia
    assert any('pneumonia' in item.title.lower() or 'chest x-ray' in item.title.lower() for item in next_steps.items)
    assert any(item.action_type == "Order Test" for item in next_steps.items)
    assert any(item.action_type == "Start Treatment" for item in next_steps.items)
    assert "pneumonia" in next_steps.patient_instructions.lower() or "breathing" in next_steps.patient_instructions.lower()


def test_paracetamol_overdose_case():
    """Test next steps generation for paracetamol overdose."""
    generator = NextStepsGenerator()
    
    # Create mock critical/high issues (overdose)
    high_issues = [{
        'agent': 'DosageChecker',
        'message': 'Critical: Paracetamol overdose detected - 1500 mg q4h',
        'evidence': ['dosage_guideline_1'],
        'score': 9.0
    }]
    
    # Create NER result with paracetamol
    ner_result = NERResult(
        entities=[],
        raw_text='Patient taking paracetamol 1500 mg every 4 hours.',
        normalized_entities={
            'drugs': ['paracetamol'],
            'dosages': [{'drug': 'paracetamol', 'dose': '1500 mg', 'frequency': 'q4h'}],
            'symptoms': []
        }
    )
    
    agent_outputs = [
        AgentOutput(
            agent='DosageChecker',
            status=Severity.HIGH,
            message='Critical: Paracetamol overdose detected',
            evidence=['dosage_guideline_1'],
            score=9.0
        )
    ]
    
    next_steps = generator.generate_next_steps(
        critical_issues=[],
        high_issues=high_issues,
        moderate_issues=[],
        low_issues=[],
        agent_outputs=agent_outputs,
        ner_result=ner_result,
        risk_score=9.0
    )
    
    # Assertions
    assert next_steps.urgency_level in ["immediate", "24h"]
    assert any('paracetamol' in item.title.lower() or 'stop' in item.title.lower() for item in next_steps.items)
    assert any('acetaminophen' in str(item.ordered_items).lower() or 'lft' in str(item.ordered_items).lower() for item in next_steps.items if item.ordered_items)
    assert any('nac' in str(item.treatment_recommendations).lower() or 'n-acetylcysteine' in str(item.treatment_recommendations).lower() for item in next_steps.items if item.treatment_recommendations)
    assert "er" in next_steps.patient_instructions.lower() or "emergency" in next_steps.patient_instructions.lower()


def test_healthy_case():
    """Test next steps generation for healthy patient."""
    generator = NextStepsGenerator()
    
    ner_result = NERResult(
        entities=[],
        raw_text='Patient is healthy with no complaints. All vitals normal.',
        normalized_entities={
            'symptoms': [],
            'vitals': [],
            'drugs': []
        }
    )
    
    next_steps = generator.generate_next_steps(
        critical_issues=[],
        high_issues=[],
        moderate_issues=[],
        low_issues=[],
        agent_outputs=[],
        ner_result=ner_result,
        risk_score=0.0
    )
    
    # Assertions
    assert next_steps.urgency_level == "routine"
    assert len(next_steps.items) >= 1
    assert "routine" in next_steps.summary.lower() or "no urgent" in next_steps.summary.lower()
    assert "routine" in next_steps.patient_instructions.lower() or "no urgent" in next_steps.patient_instructions.lower()


def test_next_steps_structure():
    """Test that next_steps object has correct structure."""
    generator = NextStepsGenerator()
    
    next_steps = generator.generate_next_steps(
        critical_issues=[],
        high_issues=[],
        moderate_issues=[],
        low_issues=[],
        agent_outputs=[],
        ner_result=None,
        risk_score=0.0
    )
    
    # Check required fields
    assert hasattr(next_steps, 'summary')
    assert hasattr(next_steps, 'urgency_level')
    assert hasattr(next_steps, 'items')
    assert hasattr(next_steps, 'patient_instructions')
    assert hasattr(next_steps, 'clinician_note')
    assert hasattr(next_steps, 'disclaimer')
    
    # Check urgency level is valid
    assert next_steps.urgency_level in ["immediate", "24h", "72h", "routine"]
    
    # Check items structure
    for item in next_steps.items:
        assert hasattr(item, 'title')
        assert hasattr(item, 'priority')
        assert hasattr(item, 'action_type')
        assert hasattr(item, 'recommended_by_agent')
        assert hasattr(item, 'rationale')
        assert hasattr(item, 'disposition')
        assert hasattr(item, 'clinical_confidence')
        assert item.priority in ["urgent", "high", "medium", "low"]
        assert item.clinical_confidence in ["high", "medium", "low"]


def test_evidence_ids_included():
    """Test that evidence_ids are included in next steps items."""
    generator = NextStepsGenerator()
    
    high_issues = [{
        'agent': 'RedFlagChecker',
        'message': 'High priority finding',
        'evidence': ['evidence_chunk_1', 'evidence_chunk_2'],
        'score': 7.0
    }]
    
    agent_outputs = [
        AgentOutput(
            agent='RedFlagChecker',
            status=Severity.HIGH,
            message='High priority finding',
            evidence=['evidence_chunk_1', 'evidence_chunk_2'],
            score=7.0
        )
    ]
    
    next_steps = generator.generate_next_steps(
        critical_issues=[],
        high_issues=high_issues,
        moderate_issues=[],
        low_issues=[],
        agent_outputs=agent_outputs,
        ner_result=None,
        risk_score=7.0
    )
    
    # Check that at least one item has evidence_ids
    items_with_evidence = [item for item in next_steps.items if item.evidence_ids]
    assert len(items_with_evidence) > 0, "At least one item should have evidence_ids"
    
    # Check evidence_ids are lists of strings
    for item in items_with_evidence:
        assert isinstance(item.evidence_ids, list)
        assert all(isinstance(eid, str) for eid in item.evidence_ids)


def test_critical_urgency_mapping():
    """Test that critical issues map to immediate urgency."""
    generator = NextStepsGenerator()
    
    critical_issues = [{
        'agent': 'RedFlagChecker',
        'message': 'Critical red flag',
        'evidence': [],
        'score': 10.0
    }]
    
    next_steps = generator.generate_next_steps(
        critical_issues=critical_issues,
        high_issues=[],
        moderate_issues=[],
        low_issues=[],
        agent_outputs=[],
        ner_result=None,
        risk_score=10.0
    )
    
    assert next_steps.urgency_level == "immediate"


def test_human_approval_flags():
    """Test that critical medications have human_approval_required flag."""
    generator = NextStepsGenerator()
    
    # Test with paracetamol overdose (should trigger NAC recommendation)
    high_issues = [{
        'agent': 'DosageChecker',
        'message': 'Paracetamol overdose',
        'evidence': [],
        'score': 9.0
    }]
    
    ner_result = NERResult(
        entities=[],
        raw_text='Paracetamol overdose',
        normalized_entities={
            'drugs': ['paracetamol'],
            'symptoms': []
        }
    )
    
    agent_outputs = [
        AgentOutput(
            agent='DosageChecker',
            status=Severity.HIGH,
            message='Paracetamol overdose',
            evidence=[],
            score=9.0
        )
    ]
    
    next_steps = generator.generate_next_steps(
        critical_issues=[],
        high_issues=high_issues,
        moderate_issues=[],
        low_issues=[],
        agent_outputs=agent_outputs,
        ner_result=ner_result,
        risk_score=9.0
    )
    
    # Check for treatment recommendations with human_approval_required
    items_with_treatment = [item for item in next_steps.items if item.treatment_recommendations]
    if items_with_treatment:
        for item in items_with_treatment:
            for treatment in item.treatment_recommendations:
                if 'nac' in treatment.drug.lower() or 'n-acetylcysteine' in treatment.drug.lower():
                    assert treatment.human_approval_required, "NAC should require human approval"

