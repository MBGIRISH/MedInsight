# Agent System Design

## Overview

MedInsight uses a multi-agent system where each agent specializes in a specific aspect of medical audit. All agents work independently and their outputs are merged by the Decision Engine.

## Agent Architecture

Each agent follows this structure:

```python
class Agent:
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
    
    def check(self, inputs) -> AgentOutput:
        # Analysis logic
        return AgentOutput(
            agent="AgentName",
            status=Severity,
            message=str,
            evidence=List[str],
            score=float,
            details=Dict
        )
```

## Agent 1: DosageCheckerAgent

### Purpose
Validates medication dosages against established guidelines and maximum safe limits.

### Inputs
- `drugs`: List of normalized drug entities
- `dosages`: List of normalized dosage entities
- `frequencies`: List of normalized frequency entities

### Logic
1. Calculate daily dose (dosage × frequency)
2. Retrieve dosage guidelines from RAG
3. Compare against maximum safe dosages
4. Flag overdoses (>max) or underdoses (<10% of max)

### Output
- **Status**: critical/high/moderate/ok
- **Score**: 0-10 (higher for more severe issues)
- **Evidence**: Specific dosage violations

### Example
```python
agent = DosageCheckerAgent(rag_service)
output = agent.check(drugs, dosages, frequencies)
# Output: "Critical: 1 overdose(s) detected"
```

## Agent 2: InteractionCheckerAgent

### Purpose
Detects dangerous drug-drug interactions.

### Inputs
- `drugs`: List of normalized drug entities

### Logic
1. Check all drug pairs
2. Query RAG for interaction information
3. Check against known dangerous interactions
4. Flag critical interactions

### Output
- **Status**: critical/ok
- **Score**: 10.0 for critical interactions
- **Evidence**: Interaction details and sources

### Example
```python
agent = InteractionCheckerAgent(rag_service)
output = agent.check(drugs)
# Output: "Critical: 1 dangerous drug interaction(s) detected"
```

## Agent 3: RedFlagCheckerAgent

### Purpose
Identifies emergency symptoms requiring immediate medical attention.

### Inputs
- `symptoms`: List of extracted symptoms
- `text`: Full document text

### Logic
1. Check symptoms against red flag list
2. Query RAG for emergency symptom information
3. Flag critical symptoms

### Output
- **Status**: critical/ok
- **Score**: 10.0 for red flags
- **Evidence**: Red flag symptoms and sources

### Example
```python
agent = RedFlagCheckerAgent(rag_service)
output = agent.check(symptoms, text)
# Output: "Critical: 1 red flag symptom(s) requiring immediate attention"
```

## Agent 4: MissingTestsCheckerAgent

### Purpose
Identifies missing essential laboratory tests based on medications prescribed.

### Inputs
- `drugs`: List of normalized drug entities
- `lab_values`: List of normalized lab test entities

### Logic
1. Map drugs to required tests
2. Check if required tests were performed
3. Flag missing essential tests

### Output
- **Status**: high/moderate/ok
- **Score**: 7.0 for high priority, 5.0 for moderate
- **Evidence**: Missing tests and associated drugs

### Example
```python
agent = MissingTestsCheckerAgent(rag_service)
output = agent.check(drugs, lab_values)
# Output: "High: 1 essential test(s) missing"
```

## Agent 5: GuidelineComplianceCheckerAgent

### Purpose
Validates treatment against clinical guidelines (WHO, NHS, etc.).

### Inputs
- `drugs`: List of normalized drug entities
- `symptoms`: List of extracted symptoms
- `text`: Full document text

### Logic
1. Extract conditions from symptoms/text
2. Retrieve treatment guidelines from RAG
3. Compare prescribed drugs with recommended treatments
4. Flag guideline violations

### Output
- **Status**: moderate/ok
- **Score**: 5.0 for violations
- **Evidence**: Guideline mismatches

### Example
```python
agent = GuidelineComplianceCheckerAgent(rag_service)
output = agent.check(drugs, symptoms, text)
# Output: "Moderate: 1 guideline compliance issue(s)"
```

## Severity Levels

- **CRITICAL**: Immediate action required (risk score 8-10)
- **HIGH**: Significant concern (risk score 6-8)
- **MODERATE**: Minor concern (risk score 4-6)
- **LOW**: Minimal concern (risk score 2-4)
- **OK**: No issues detected (risk score 0-2)

## Agent Output Schema

```python
{
    "agent": "AgentName",
    "status": "critical|high|moderate|low|ok",
    "message": "Human-readable message",
    "evidence": ["Evidence string 1", "Evidence string 2"],
    "score": 0.0-10.0,
    "details": {
        "issues": [...],
        "additional_data": ...
    }
}
```

## Decision Engine Integration

The Decision Engine:
1. Collects all agent outputs
2. Categorizes by severity
3. Calculates weighted risk score
4. Generates recommendations
5. Creates final audit report

## Extensibility

New agents can be added by:
1. Creating a new agent class inheriting the pattern
2. Implementing `check()` method
3. Returning `AgentOutput`
4. Registering in the audit pipeline

