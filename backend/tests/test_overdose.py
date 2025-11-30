"""
Test case for overdose detection.
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ner import NERService
from app.services.normalizer import NormalizerService
from app.services.rag import RAGService
from app.services.agents import DosageCheckerAgent
import json


def load_sample_case(case_name: str) -> str:
    """Load sample case from JSON."""
    cases_path = Path(__file__).parent / "sample_cases.json"
    with open(cases_path, 'r') as f:
        cases = json.load(f)
    return cases.get(case_name, {}).get('text', '')


def test_overdose_detection():
    """Test that overdose is detected correctly."""
    # Load overdose case
    text = load_sample_case("overdose_case")
    assert text, "Overdose case text not found"
    
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
    assert normalized.get('dosages'), "No dosages found"
    
    # Check dosage
    agent = DosageCheckerAgent(rag_service)
    output = agent.check(
        normalized.get('drugs', []),
        normalized.get('dosages', []),
        normalized.get('frequencies', [])
    )
    
    # Assert overdose detected
    assert output.status.value in ['critical', 'high'], \
        f"Expected critical/high status, got {output.status.value}"
    assert output.score > 0, "Expected non-zero score for overdose"
    assert "overdose" in output.message.lower() or "dosage" in output.message.lower(), \
        "Expected overdose message"
    
    print(f"✅ Overdose test passed: {output.message}")


if __name__ == "__main__":
    test_overdose_detection()

