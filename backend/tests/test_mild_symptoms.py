"""
Unit tests for mild symptoms handling.
"""
import pytest
from app.services.agents import RedFlagCheckerAgent
from app.services.decision_engine import DecisionEngine
from app.services.rag import RAGService
from app.services.llm_service import LLMService
from app.models.schemas import AgentOutput, Severity, NERResult


@pytest.fixture
def red_flag_agent():
    """Create RedFlagCheckerAgent instance."""
    rag = RAGService()
    llm = LLMService()
    return RedFlagCheckerAgent(rag, llm)


def test_mild_symptoms_only_returns_ok(red_flag_agent):
    """Test that only mild symptoms return OK status."""
    mild_symptoms = ['runny nose', 'sneezing', 'watery eyes']
    result = red_flag_agent.check(mild_symptoms)
    
    assert result.status == Severity.OK, "Mild symptoms should return OK"
    assert result.score == 0.0, "Mild symptoms should have score 0"
    assert "Mild self-limiting symptoms" in result.message or "No medical risk" in result.message


def test_mild_symptoms_with_fever_returns_severity(red_flag_agent):
    """Test that mild symptoms with fever still trigger severity."""
    symptoms_with_fever = ['runny nose', 'sneezing', 'fever']
    result = red_flag_agent.check(symptoms_with_fever)
    
    assert result.status != Severity.OK, "Symptoms with fever should NOT return OK"
    assert result.score > 0.0, "Symptoms with fever should have score > 0"


def test_mild_symptoms_with_chest_pain_returns_severity(red_flag_agent):
    """Test that mild symptoms with chest pain still trigger severity."""
    symptoms_with_chest_pain = ['runny nose', 'chest pain']
    result = red_flag_agent.check(symptoms_with_chest_pain)
    
    assert result.status != Severity.OK, "Symptoms with chest pain should NOT return OK"
    assert result.score > 0.0, "Symptoms with chest pain should have score > 0"


def test_all_agents_ok_forces_score_zero():
    """Test that if all agents are OK, final score is 0 regardless of symptoms."""
    agent_outputs = [
        AgentOutput(agent='RedFlagChecker', status=Severity.OK, message='Mild symptoms', evidence=[], score=0.0),
        AgentOutput(agent='DosageChecker', status=Severity.OK, message='No medications', evidence=[], score=0.0),
        AgentOutput(agent='InteractionChecker', status=Severity.OK, message='No interactions', evidence=[], score=0.0),
    ]
    
    ner_result = NERResult(
        entities=[],
        raw_text='Patient has runny nose and sneezing.',
        normalized_entities={
            'symptoms': ['runny nose', 'sneezing'],
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
    
    assert report.risk_score == 0.0, "All agents OK should force score to 0"
    assert len(report.critical_issues) == 0, "No critical issues when all agents OK"
    assert len(report.high_issues) == 0, "No high issues when all agents OK"


def test_mild_cough_without_fever_returns_ok(red_flag_agent):
    """Test that mild cough without fever returns OK."""
    mild_symptoms = ['mild cough', 'runny nose']
    result = red_flag_agent.check(mild_symptoms)
    
    assert result.status == Severity.OK, "Mild cough without fever should return OK"
    assert result.score == 0.0, "Mild cough without fever should have score 0"


def test_seasonal_allergy_symptoms_return_ok(red_flag_agent):
    """Test that seasonal allergy symptoms return OK."""
    allergy_symptoms = ['seasonal allergy', 'itchy nose', 'sneezing', 'watery eyes']
    result = red_flag_agent.check(allergy_symptoms)
    
    assert result.status == Severity.OK, "Seasonal allergy symptoms should return OK"
    assert result.score == 0.0, "Seasonal allergy symptoms should have score 0"

