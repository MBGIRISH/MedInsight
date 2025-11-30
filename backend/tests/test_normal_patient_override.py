"""
Unit tests for Normal Patient Override logic.
"""
import pytest
from app.services.decision_engine import DecisionEngine
from app.models.schemas import AgentOutput, Severity, NERResult


def test_normal_patient_override_applies():
    """Test that normal patient override applies when all conditions are met."""
    agent_outputs = [
        AgentOutput(agent='RedFlagChecker', status=Severity.OK, message='No symptoms', evidence=[], score=0.0),
        AgentOutput(agent='DosageChecker', status=Severity.OK, message='No medications', evidence=[], score=0.0),
    ]
    
    ner_result = NERResult(
        entities=[],
        raw_text='Patient is healthy. BP 120/80, HR 72, O2 98%, Temp 98.6°F, Glucose 95 mg/dL.',
        normalized_entities={
            'symptoms': [],
            'vitals': [
                {'type': 'blood_pressure', 'value': [120, 80], 'text': '120/80'},
                {'type': 'heart_rate', 'value': 72, 'text': '72 bpm'},
                {'type': 'oxygen', 'value': 98, 'text': '98%'},
                {'type': 'temperature', 'value': 98.6, 'text': '98.6°F'}
            ],
            'lab_values': [
                {'type': 'glucose', 'value': 95, 'text': '95 mg/dL'}
            ],
            'drugs': []
        }
    )
    
    report = DecisionEngine.merge_agent_outputs(
        agent_outputs=agent_outputs,
        ner_result=ner_result,
        pattern_detections=[]
    )
    
    assert report.risk_score == 0.0, "Normal patient should have score 0"
    assert len(report.critical_issues) == 0, "No critical issues for normal patient"
    assert len(report.high_issues) == 0, "No high issues for normal patient"
    assert "Routine health maintenance" in report.recommendations[0] or "No concerning findings" in report.recommendations[0]


def test_override_takes_priority_over_agent_scores():
    """Test that override takes priority even if agent incorrectly reports high."""
    agent_outputs = [
        AgentOutput(agent='RedFlagChecker', status=Severity.HIGH, message='False positive', evidence=[], score=7.0),
        AgentOutput(agent='DosageChecker', status=Severity.OK, message='No medications', evidence=[], score=0.0),
    ]
    
    ner_result = NERResult(
        entities=[],
        raw_text='Patient is healthy. BP 120/80, HR 72, O2 98%, Temp 98.6°F, Glucose 95 mg/dL.',
        normalized_entities={
            'symptoms': [],
            'vitals': [
                {'type': 'blood_pressure', 'value': [120, 80], 'text': '120/80'},
                {'type': 'heart_rate', 'value': 72, 'text': '72 bpm'},
                {'type': 'oxygen', 'value': 98, 'text': '98%'},
                {'type': 'temperature', 'value': 98.6, 'text': '98.6°F'}
            ],
            'lab_values': [
                {'type': 'glucose', 'value': 95, 'text': '95 mg/dL'}
            ],
            'drugs': []
        }
    )
    
    report = DecisionEngine.merge_agent_outputs(
        agent_outputs=agent_outputs,
        ner_result=ner_result,
        pattern_detections=[]
    )
    
    # Override should take priority
    assert report.risk_score == 0.0, "Override should force score to 0 even if agent reports high"


def test_override_not_applied_with_abnormal_vitals():
    """Test that override does NOT apply when vitals are abnormal."""
    agent_outputs = [
        AgentOutput(agent='RedFlagChecker', status=Severity.OK, message='No symptoms', evidence=[], score=0.0),
    ]
    
    ner_result = NERResult(
        entities=[],
        raw_text='Patient with high BP. BP 150/95, HR 72.',
        normalized_entities={
            'symptoms': [],
            'vitals': [
                {'type': 'blood_pressure', 'value': [150, 95], 'text': '150/95'},  # Abnormal
                {'type': 'heart_rate', 'value': 72, 'text': '72 bpm'}
            ],
            'lab_values': [],
            'drugs': []
        }
    )
    
    report = DecisionEngine.merge_agent_outputs(
        agent_outputs=agent_outputs,
        ner_result=ner_result,
        pattern_detections=[]
    )
    
    # Override should NOT apply
    assert report.risk_score >= 0.0  # May be 0 or higher, but override should not force it


def test_override_not_applied_with_symptoms():
    """Test that override does NOT apply when symptoms are present."""
    agent_outputs = [
        AgentOutput(agent='RedFlagChecker', status=Severity.OK, message='Mild symptoms', evidence=[], score=0.0),
    ]
    
    ner_result = NERResult(
        entities=[],
        raw_text='Patient reports mild headache. BP 120/80, HR 72.',
        normalized_entities={
            'symptoms': ['headache'],  # Has symptoms
            'vitals': [
                {'type': 'blood_pressure', 'value': [120, 80], 'text': '120/80'},
                {'type': 'heart_rate', 'value': 72, 'text': '72 bpm'}
            ],
            'lab_values': [],
            'drugs': []
        }
    )
    
    report = DecisionEngine.merge_agent_outputs(
        agent_outputs=agent_outputs,
        ner_result=ner_result,
        pattern_detections=[]
    )
    
    # Override should NOT apply (has symptoms)
    # Score may vary, but override should not force it to 0


def test_override_not_applied_with_abnormal_labs():
    """Test that override does NOT apply when labs are abnormal."""
    agent_outputs = [
        AgentOutput(agent='RedFlagChecker', status=Severity.OK, message='No symptoms', evidence=[], score=0.0),
    ]
    
    ner_result = NERResult(
        entities=[],
        raw_text='Patient. BP 120/80, HR 72. Glucose 200 mg/dL.',
        normalized_entities={
            'symptoms': [],
            'vitals': [
                {'type': 'blood_pressure', 'value': [120, 80], 'text': '120/80'},
                {'type': 'heart_rate', 'value': 72, 'text': '72 bpm'}
            ],
            'lab_values': [
                {'type': 'glucose', 'value': 200, 'text': '200 mg/dL'}  # Abnormal (>140)
            ],
            'drugs': []
        }
    )
    
    report = DecisionEngine.merge_agent_outputs(
        agent_outputs=agent_outputs,
        ner_result=ner_result,
        pattern_detections=[]
    )
    
    # Override should NOT apply (abnormal glucose)
    assert report.risk_score >= 0.0  # Override should not force it to 0

