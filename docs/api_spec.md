# MedInsight API Specification

## Base URL
```
http://localhost:8000
```

## Endpoints

### 1. Health Check

**GET** `/health`

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "services": {
    "api": "operational",
    "database": "operational",
    "vectorstore": "operational"
  }
}
```

---

### 2. Ingest Document

**POST** `/api/ingest`

Upload and extract text from PDF or image.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Body**: File upload (PDF or image)

**Response:**
```json
{
  "text": "Extracted text...",
  "chunks": ["chunk 1", "chunk 2"],
  "ner_result": {
    "entities": [
      {
        "text": "aspirin",
        "type": "DRUG",
        "start": 0,
        "end": 7,
        "confidence": 0.9
      }
    ],
    "raw_text": "...",
    "normalized_entities": {}
  }
}
```

**Status Codes:**
- `200`: Success
- `400`: Bad request (unsupported file type)
- `500`: Server error

---

### 3. Run Audit

**POST** `/api/audit`

Run complete audit pipeline on uploaded document.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Body**: File upload (PDF or image)

**Alternative Request (JSON):**
```json
{
  "text": "Clinical text...",
  "file_path": "/path/to/file.pdf"
}
```

**Response:**
```json
{
  "audit_id": "uuid-string",
  "timestamp": "2024-01-01T12:00:00",
  "critical_issues": [
    {
      "agent": "DosageChecker",
      "message": "Critical: 1 overdose(s) detected",
      "evidence": ["..."],
      "details": {...},
      "score": 10.0
    }
  ],
  "high_issues": [],
  "moderate_issues": [],
  "safe_items": [],
  "recommendations": [
    "URGENT: Review and adjust medication dosages..."
  ],
  "risk_score": 8.5,
  "agent_outputs": [...],
  "ner_result": {...}
}
```

**Status Codes:**
- `200`: Success
- `400`: Bad request (no text/file provided)
- `404`: File not found (if using file_path)
- `500`: Server error

---

### 4. Get Audit

**GET** `/api/audit/{audit_id}`

Retrieve a past audit by ID.

**Response:**
Same as `/api/audit` response.

**Status Codes:**
- `200`: Success
- `404`: Audit not found

---

## Error Response Format

```json
{
  "detail": "Error message"
}
```

## Example Usage

### Python
```python
import requests

# Upload and audit
with open("prescription.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/api/audit",
        files=files
    )
    audit = response.json()
    print(f"Risk Score: {audit['risk_score']}")
```

### cURL
```bash
curl -X POST "http://localhost:8000/api/audit" \
  -F "file=@prescription.pdf"
```

### JavaScript
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/api/audit', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## Rate Limiting

Currently no rate limiting implemented. For production, consider:
- Rate limiting per IP
- Authentication/authorization
- Request size limits

## Authentication

Currently no authentication required. For production, add:
- API keys
- JWT tokens
- OAuth2

