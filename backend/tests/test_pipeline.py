"""
Integration test for complete audit pipeline.
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
from app.services.decision_engine import DecisionEngine
import json


def load_sample_case(case_name: str) -> str:
    """Load sample case from JSON."""
    cases_path = Path(__file__).parent / "sample_cases.json"
    with open(cases_path, 'r') as f:
        cases = json.load(f)
    return cases.get(case_name, {}).get('text', '')


def test_complete_pipeline():
    """Test complete audit pipeline end-to-end."""
    # Use interaction case as it has multiple issues
    text = load_sample_case("interaction_case")
    assert text, "Sample case text not found"
    
    # Step 1: NER
    ner_service = NERService()
    ner_result = ner_service.extract_entities(text)
    assert ner_result.entities, "NER failed - no entities extracted"
    
    # Step 2: Normalize
    normalizer_service = NormalizerService()
    normalized = normalizer_service.normalize_entities(ner_result.entities, text)
    assert normalized, "Normalization failed"
    
    # Step 3: Initialize agents
    rag_service = RAGService()
    agents = [
        DosageCheckerAgent(rag_service),
        InteractionCheckerAgent(rag_service),
        RedFlagCheckerAgent(rag_service),
        MissingTestsCheckerAgent(rag_service),
        GuidelineComplianceCheckerAgent(rag_service)
    ]
    
    # Step 4: Run all agents
    agent_outputs = []
    agent_outputs.append(
        agents[0].check(
            normalized.get('drugs', []),
            normalized.get('dosages', []),
            normalized.get('frequencies', [])
        )
    )
    agent_outputs.append(agents[1].check(normalized.get('drugs', [])))
    agent_outputs.append(
        agents[2].check(normalized.get('symptoms', []), text)
    )
    agent_outputs.append(
        agents[3].check(
            normalized.get('drugs', []),
            normalized.get('lab_values', [])
        )
    )
    agent_outputs.append(
        agents[4].check(
            normalized.get('drugs', []),
            normalized.get('symptoms', []),
            text
        )
    )
    
    assert len(agent_outputs) == 5, "Not all agents ran"
    
    # Step 5: Decision engine
    audit_report = DecisionEngine.merge_agent_outputs(agent_outputs, ner_result)
    
    assert audit_report.risk_score >= 0, "Risk score should be >= 0"
    assert audit_report.risk_score <= 10, "Risk score should be <= 10"
    assert audit_report.audit_id, "Audit ID should be generated"
    assert len(audit_report.agent_outputs) == 5, "All agent outputs should be included"
    
    print(f"✅ Pipeline test passed!")
    print(f"   Risk Score: {audit_report.risk_score:.2f}/10")
    print(f"   Critical Issues: {len(audit_report.critical_issues)}")
    print(f"   High Issues: {len(audit_report.high_issues)}")
    print(f"   Recommendations: {len(audit_report.recommendations)}")


if __name__ == "__main__":
    test_complete_pipeline()

