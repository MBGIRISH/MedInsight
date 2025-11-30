"""
Test case for missing essential tests detection.
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ner import NERService
from app.services.normalizer import NormalizerService
from app.services.rag import RAGService
from app.services.agents import MissingTestsCheckerAgent
import json


def load_sample_case(case_name: str) -> str:
    """Load sample case from JSON."""
    cases_path = Path(__file__).parent / "sample_cases.json"
    with open(cases_path, 'r') as f:
        cases = json.load(f)
    return cases.get(case_name, {}).get('text', '')


def test_missing_tests_detection():
    """Test that missing essential tests are detected correctly."""
    # Load missing tests case
    text = load_sample_case("missing_tests_case")
    assert text, "Missing tests case text not found"
    
    # Initialize services
    ner_service = NERService()
    normalizer_service = NormalizerService()
    rag_service = RAGService()
    
    # Extract entities
    ner_result = ner_service.extract_entities(text)
    assert ner_result.entities, "No entities extracted"
    
    # Normalize
    normalized = normalizer_service.normalize_entities(ner_result.entities, text)
    assert normalized.get('drugs'), "No drugs found"
    
    # Check missing tests (atorvastatin requires liver function tests)
    agent = MissingTestsCheckerAgent(rag_service)
    output = agent.check(
        normalized.get('drugs', []),
        normalized.get('lab_values', [])
    )
    
    # Assert missing tests detected
    assert output.status.value in ['critical', 'high', 'moderate'], \
        f"Expected critical/high/moderate status, got {output.status.value}"
    assert "test" in output.message.lower() or "missing" in output.message.lower(), \
        "Expected missing test message"
    
    print(f"✅ Missing tests test passed: {output.message}")


if __name__ == "__main__":
    test_missing_tests_detection()

