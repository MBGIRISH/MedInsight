"""
Unit tests for 5-level clinical severity logic.
"""
import pytest
from app.services.agents import RedFlagCheckerAgent
from app.services.decision_engine import DecisionEngine
from app.services.rag import RAGService
from app.models.schemas import AgentOutput, Severity, NERResult


@pytest.fixture
def red_flag_agent():
    """Create RedFlagCheckerAgent instance."""
    rag = RAGService()
    return RedFlagCheckerAgent(rag)


def test_low_severity_mild_symptoms(red_flag_agent):
    """Test that mild symptoms return LOW severity (1-2)."""
    symptoms = ['runny nose', 'sneezing']
    result = red_flag_agent.check(symptoms, "Patient reports runny nose and sneezing. No fever.")
    
    assert result.status == Severity.LOW, "Mild symptoms should return LOW"
    assert 1.0 <= result.score <= 2.0, f"LOW severity should have score 1-2, got {result.score}"


def test_moderate_severity_fever_100(red_flag_agent):
    """Test that fever 100°F returns MODERATE severity (3-5)."""
    symptoms = ['fever']
    result = red_flag_agent.check(symptoms, "Patient reports fever of 100.5°F and sore throat.")
    
    assert result.status == Severity.MODERATE, "Fever 100°F should return MODERATE"
    assert 3.0 <= result.score <= 5.0, f"MODERATE severity should have score 3-5, got {result.score}"


def test_high_severity_fever_102(red_flag_agent):
    """Test that fever ≥101.5°F returns HIGH severity (6-8)."""
    symptoms = ['fever', 'cough']
    result = red_flag_agent.check(symptoms, "Patient reports fever of 102°F and cough with sputum.")
    
    assert result.status == Severity.HIGH, "Fever 102°F should return HIGH"
    assert 6.0 <= result.score <= 8.0, f"HIGH severity should have score 6-8, got {result.score}"


def test_critical_severity_chest_pain(red_flag_agent):
    """Test that chest pain returns CRITICAL severity (9-10)."""
    symptoms = ['chest pain', 'shortness of breath']
    result = red_flag_agent.check(symptoms, "Patient reports chest pain and shortness of breath. Oxygen saturation 90%.")
    
    assert result.status == Severity.CRITICAL, "Chest pain should return CRITICAL"
    assert 9.0 <= result.score <= 10.0, f"CRITICAL severity should have score 9-10, got {result.score}"


def test_all_agents_ok_returns_ok():
    """Test that if all agents are OK, final score is 0."""
    agent_outputs = [
        AgentOutput(agent='RedFlagChecker', status=Severity.OK, message='No symptoms', evidence=[], score=0.0),
        AgentOutput(agent='DosageChecker', status=Severity.OK, message='No medications', evidence=[], score=0.0),
    ]
    
    ner_result = NERResult(
        entities=[],
        raw_text='Patient is healthy.',
        normalized_entities={
            'symptoms': [],
            'vitals': [],
            'lab_values': [],
            'drugs': []
        }
    )
    
    report = DecisionEngine.merge_agent_outputs(
        agent_outputs=agent_outputs,
        ner_result=ner_result,
        pattern_detections=[]
    )
    
    assert report.risk_score == 0.0, "All agents OK should return score 0"


def test_all_agents_low_returns_low():
    """Test that if all agents are OK or LOW, final severity is LOW."""
    agent_outputs = [
        AgentOutput(agent='RedFlagChecker', status=Severity.LOW, message='Mild symptoms', evidence=[], score=1.5),
        AgentOutput(agent='DosageChecker', status=Severity.OK, message='No medications', evidence=[], score=0.0),
    ]
    
    ner_result = NERResult(
        entities=[],
        raw_text='Patient reports runny nose.',
        normalized_entities={
            'symptoms': ['runny nose'],
            'vitals': [],
            'lab_values': [],
            'drugs': []
        }
    )
    
    report = DecisionEngine.merge_agent_outputs(
        agent_outputs=agent_outputs,
        ner_result=ner_result,
        pattern_detections=[]
    )
    
    assert 1.0 <= report.risk_score <= 2.0, f"All agents OK/LOW should return score 1-2, got {report.risk_score}"

