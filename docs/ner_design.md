# Named Entity Recognition (NER) Design

## Overview

The NER service extracts medical entities from clinical text using a combination of transformer models and rule-based patterns.

## Entity Types

1. **DRUG**: Medication names
2. **DOSAGE**: Medication dosages (e.g., "500 mg", "2 tablets")
3. **FREQUENCY**: Dosing frequency (e.g., "bid", "twice daily")
4. **DURATION**: Treatment duration (e.g., "7 days", "2 weeks")
5. **SYMPTOM**: Patient symptoms
6. **LAB_VALUE**: Laboratory test results
7. **VITALS**: Vital signs (e.g., blood pressure)

## Architecture

### Hybrid Approach

The NER service uses a two-stage approach:

1. **Model-based Extraction**: HuggingFace BERT NER model
2. **Rule-based Extraction**: Pattern matching for medical terms

### Model-based Extraction

- **Model**: `dslim/bert-base-NER`
- **Pipeline**: HuggingFace transformers pipeline
- **Aggregation**: Simple aggregation strategy
- **Fallback**: If model fails, falls back to rule-based only

### Rule-based Patterns

#### Drug Patterns
```python
drug_patterns = [
    r'\b(aspirin|ibuprofen|paracetamol|...)\b',  # Common drugs
    r'\b([A-Z][a-z]+(?:pril|statin|mycin|...))\b'  # Drug suffixes
]
```

#### Dosage Patterns
```python
dosage_patterns = [
    r'\b(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg|units?|tablets?)\b',
    r'\b(\d+/\d+)\s*(mg|g|ml|tablet|cap)\b',
    r'\b(one|two|three|half|quarter)\s*(tablet|cap|pill|mg|g)\b'
]
```

#### Frequency Patterns
```python
freq_patterns = [
    r'\b(qd|bid|tid|qid|once|twice|daily|weekly|monthly)\b',
    r'\b(\d+)\s*(times?|x)\s*(per|a|daily|week|day)\b'
]
```

#### Lab Value Patterns
```python
lab_patterns = [
    r'\b(glucose|creatinine|hemoglobin|...)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(mg/dl|mmol/l|units?)\b',
    r'\b(bp|blood pressure)\s*[:=]?\s*(\d+/\d+)\b'
]
```

## Normalization

After extraction, entities are normalized:

### Dosage Normalization
- Convert all units to mg
- Handle fractions (½ tablet → 0.5 × tablet_mg)
- Convert word-based dosages ("two tablets" → 2 × tablet_mg)

### Frequency Normalization
- Standardize abbreviations (qd → once daily)
- Extract times per day

### Duration Normalization
- Convert to days
- Calculate weeks and months

### Lab Value Normalization
- Extract test name and value
- Preserve units

## Output Format

```python
{
    "entities": [
        {
            "text": "aspirin",
            "type": "DRUG",
            "start": 0,
            "end": 7,
            "confidence": 0.9
        },
        {
            "text": "100 mg",
            "type": "DOSAGE",
            "start": 8,
            "end": 14,
            "confidence": 0.8
        }
    ],
    "raw_text": "aspirin 100 mg once daily",
    "normalized_entities": {
        "drugs": [...],
        "dosages": [...],
        "frequencies": [...],
        ...
    }
}
```

## Performance Considerations

- Rule-based patterns are fast but may have false positives
- Model-based extraction is slower but more accurate
- Combination provides best balance
- Deduplication removes overlapping entities

## Future Improvements

1. Use medical-specific NER models (e.g., scispacy)
2. Fine-tune BERT on medical text
3. Add more sophisticated pattern matching
4. Implement entity linking to medical ontologies
5. Add confidence thresholding

