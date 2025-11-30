"""
Test case for drug interaction detection.
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ner import NERService
from app.services.normalizer import NormalizerService
from app.services.rag import RAGService
from app.services.agents import InteractionCheckerAgent
import json


def load_sample_case(case_name: str) -> str:
    """Load sample case from JSON."""
    cases_path = Path(__file__).parent / "sample_cases.json"
    with open(cases_path, 'r') as f:
        cases = json.load(f)
    return cases.get(case_name, {}).get('text', '')


def test_interaction_detection():
    """Test that drug interactions are detected correctly."""
    # Load interaction case
    text = load_sample_case("interaction_case")
    assert text, "Interaction case text not found"
    
    # Initialize services
    ner_service = NERService()
    normalizer_service = NormalizerService()
    rag_service = RAGService()
    
    # Extract entities
    ner_result = ner_service.extract_entities(text)
    assert ner_result.entities, "No entities extracted"
    
    # Normalize
    normalized = normalizer_service.normalize_entities(ner_result.entities, text)
    assert len(normalized.get('drugs', [])) >= 2, "Need at least 2 drugs for interaction test"
    
    # Check interactions
    agent = InteractionCheckerAgent(rag_service)
    output = agent.check(normalized.get('drugs', []))
    
    # Assert interaction detected (warfarin + aspirin is dangerous)
    assert output.status.value in ['critical', 'high'], \
        f"Expected critical/high status, got {output.status.value}"
    assert output.score > 0, "Expected non-zero score for interaction"
    assert "interaction" in output.message.lower(), \
        "Expected interaction message"
    
    print(f"✅ Interaction test passed: {output.message}")


if __name__ == "__main__":
    test_interaction_detection()

