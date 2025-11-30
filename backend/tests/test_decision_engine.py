"""
Unit tests for DecisionEngine merge logic.
"""
import pytest
from app.services.decision_engine import DecisionEngine
from app.models.schemas import AgentOutput, Severity, NERResult


def test_no_agents_returns_ok():
    """Test that empty agent list returns OK with score 0."""
    ner_result = NERResult(entities=[], raw_text='test', normalized_entities={})
    result = DecisionEngine.merge_agent_outputs([], ner_result, pattern_detections=[])
    assert result.risk_score == 0.0
    assert len(result.critical_issues) == 0
    assert len(result.high_issues) == 0
    assert len(result.moderate_issues) == 0
    assert len(result.safe_items) == 0


def test_single_critical_overrides():
    """Test that critical severity overrides high."""
    agents = [
        AgentOutput(agent="DosageChecker", status=Severity.HIGH, message="overdose", score=7.0, evidence=[]),
        AgentOutput(agent="RedFlagChecker", status=Severity.CRITICAL, message="meningitis", score=10.0, evidence=[])
    ]
    ner_result = NERResult(entities=[], raw_text='test', normalized_entities={})
    result = DecisionEngine.merge_agent_outputs(agents, ner_result, pattern_detections=[])
    assert result.risk_score >= 9.0
    assert result.risk_score <= 10.0
    assert len(result.critical_issues) > 0


def test_multiple_high_pick_max():
    """Test that multiple high agents use max score."""
    agents = [
        AgentOutput(agent="A", status=Severity.HIGH, message="", score=6.0, evidence=[]),
        AgentOutput(agent="B", status=Severity.HIGH, message="", score=8.0, evidence=[])
    ]
    ner_result = NERResult(entities=[], raw_text='test', normalized_entities={})
    result = DecisionEngine.merge_agent_outputs(agents, ner_result, pattern_detections=[])
    assert result.risk_score == 8.0
    assert len(result.high_issues) == 2


def test_normal_case_safe():
    """Test that healthy case returns OK with score 0."""
    agents = [
        AgentOutput(agent="RedFlagChecker", status=Severity.OK, message="", score=0.0, evidence=[]),
        AgentOutput(agent="DosageChecker", status=Severity.OK, message="", score=0.0, evidence=[])
    ]
    ner_result = NERResult(entities=[], raw_text='test', normalized_entities={})
    result = DecisionEngine.merge_agent_outputs(agents, ner_result, pattern_detections=[])
    assert result.risk_score == 0.0
    assert len(result.safe_items) >= 2


def test_score_clamping():
    """Test that scores are clamped to severity ranges."""
    # Test critical score clamping
    agents = [
        AgentOutput(agent="A", status=Severity.CRITICAL, message="", score=15.0, evidence=[])  # Should clamp to 10
    ]
    ner_result = NERResult(entities=[], raw_text='test', normalized_entities={})
    result = DecisionEngine.merge_agent_outputs(agents, ner_result, pattern_detections=[])
    assert result.risk_score <= 10.0
    assert result.risk_score >= 9.0
    
    # Test high score clamping
    agents2 = [
        AgentOutput(agent="A", status=Severity.HIGH, message="", score=10.0, evidence=[])  # Should clamp to 8
    ]
    result2 = DecisionEngine.merge_agent_outputs(agents2, ner_result, pattern_detections=[])
    assert result2.risk_score <= 8.0
    assert result2.risk_score >= 6.0


def test_moderate_score_range():
    """Test that moderate scores are in 3-5 range."""
    agents = [
        AgentOutput(agent="A", status=Severity.MODERATE, message="", score=4.0, evidence=[])
    ]
    ner_result = NERResult(entities=[], raw_text='test', normalized_entities={})
    result = DecisionEngine.merge_agent_outputs(agents, ner_result, pattern_detections=[])
    assert result.risk_score >= 3.0
    assert result.risk_score <= 5.0


def test_low_score_range():
    """Test that low scores are in 1-2 range."""
    agents = [
        AgentOutput(agent="A", status=Severity.LOW, message="", score=1.5, evidence=[])
    ]
    ner_result = NERResult(entities=[], raw_text='test', normalized_entities={})
    result = DecisionEngine.merge_agent_outputs(agents, ner_result, pattern_detections=[])
    assert result.risk_score >= 1.0
    assert result.risk_score <= 2.0


def test_safety_override():
    """Test safety override when no symptoms or vitals AND all agents OK."""
    # Safety override only applies when all agents return OK
    agents = [
        AgentOutput(agent="A", status=Severity.OK, message="", score=0.0, evidence=[]),
        AgentOutput(agent="B", status=Severity.OK, message="", score=0.0, evidence=[])
    ]
    ner_result = NERResult(
        entities=[],
        raw_text='Patient is healthy',
        normalized_entities={'symptoms': [], 'vitals': [], 'drugs': []}
    )
    result = DecisionEngine.merge_agent_outputs(agents, ner_result, pattern_detections=[])
    # Should confirm score 0 if no symptoms/vitals and all agents OK
    assert result.risk_score == 0.0


def test_highest_severity_wins():
    """Test that highest severity among agents is chosen."""
    agents = [
        AgentOutput(agent="A", status=Severity.LOW, message="", score=1.0, evidence=[]),
        AgentOutput(agent="B", status=Severity.MODERATE, message="", score=4.0, evidence=[]),
        AgentOutput(agent="C", status=Severity.HIGH, message="", score=7.0, evidence=[])
    ]
    ner_result = NERResult(entities=[], raw_text='test', normalized_entities={})
    result = DecisionEngine.merge_agent_outputs(agents, ner_result, pattern_detections=[])
    # Should use HIGH (7.0), not MODERATE or LOW
    assert result.risk_score >= 6.0
    assert result.risk_score <= 8.0
    assert len(result.high_issues) > 0

