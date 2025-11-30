"""
Test case for red flag symptom detection.
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ner import NERService
from app.services.normalizer import NormalizerService
from app.services.rag import RAGService
from app.services.agents import RedFlagCheckerAgent
import json


def load_sample_case(case_name: str) -> str:
    """Load sample case from JSON."""
    cases_path = Path(__file__).parent / "sample_cases.json"
    with open(cases_path, 'r') as f:
        cases = json.load(f)
    return cases.get(case_name, {}).get('text', '')


def test_redflag_detection():
    """Test that red flag symptoms are detected correctly."""
    # Load red flag case
    text = load_sample_case("redflag_case")
    assert text, "Red flag case text not found"
    
    # Initialize services
    ner_service = NERService()
    normalizer_service = NormalizerService()
    rag_service = RAGService()
    
    # Extract entities
    ner_result = ner_service.extract_entities(text)
    assert ner_result.entities, "No entities extracted"
    
    # Normalize
    normalized = normalizer_service.normalize_entities(ner_result.entities, text)
    
    # Check red flags
    agent = RedFlagCheckerAgent(rag_service)
    output = agent.check(
        normalized.get('symptoms', []),
        text
    )
    
    # Assert red flag detected (chest pain is a red flag)
    assert output.status.value in ['critical', 'high'], \
        f"Expected critical/high status, got {output.status.value}"
    assert output.score > 0, "Expected non-zero score for red flag"
    assert "red flag" in output.message.lower() or "critical" in output.message.lower(), \
        "Expected red flag message"
    
    print(f"✅ Red flag test passed: {output.message}")


if __name__ == "__main__":
    test_redflag_detection()

