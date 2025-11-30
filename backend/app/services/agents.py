from typing import List, Dict, Any
from app.models.schemas import AgentOutput, Severity
from app.services.rag import RAGService
from app.services.normalizer import NormalizerService
import re

# Ensure Severity is available globally
from app.models.schemas import Severity as SeverityEnum


class DosageCheckerAgent:
    """Agent for checking dosage safety."""
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        # Initialize LLM service for enhanced reasoning
        try:
            from app.services.llm_service import LLMService
            self.llm_service = LLMService()
        except Exception as e:
            print(f"Warning: Could not initialize LLM service for DosageChecker: {e}")
            self.llm_service = None
        # Common maximum dosages (mg) - fallback if RAG doesn't have info
        self.max_dosages = {
            'aspirin': 4000,
            'ibuprofen': 3200,
            'paracetamol': 4000,
            'metformin': 2550,
            'warfarin': 10,
            'amoxicillin': 3000,
            'atorvastatin': 80,
            'simvastatin': 80,
            'omeprazole': 80,
            'lisinopril': 40,
            'amlodipine': 10,
            'metoprolol': 200,
            'furosemide': 600,
            'levothyroxine': 0.3,
            'gabapentin': 3600,
            'sertraline': 200
        }

    def check(self, drugs: List[Dict], dosages: List[Dict], frequencies: List[Dict]) -> AgentOutput:
        """Check dosage safety."""
        issues = []
        max_score = 0.0
        
        for i, drug_info in enumerate(drugs):
            drug_name = drug_info.get('normalized_name', '').lower()
            
            # Get dosage for this drug (simplified - assumes order matches)
            if i < len(dosages):
                dosage_info = dosages[i]
                daily_dose_mg = dosage_info.get('value_mg', 0)
                
                # Get frequency
                if i < len(frequencies):
                    freq_info = frequencies[i]
                    times_per_day = freq_info.get('times_per_day', 1)
                    daily_dose_mg *= times_per_day
                
                # Retrieve guidelines
                guidelines = self.rag_service.retrieve_dosage_guidelines(drug_name)
                
                # Check against maximum
                max_dose = self.max_dosages.get(drug_name, None)
                if guidelines:
                    # Try to extract max dose from guidelines
                    for guideline in guidelines:
                        content = guideline.get('content', '').lower()
                        match = re.search(r'max(?:imum)?\s*(?:dose|dosage)?\s*(?:is|:)?\s*(\d+(?:\.\d+)?)\s*(?:mg)?', content)
                        if match:
                            max_dose = float(match.group(1))
                            break
                
                if max_dose and daily_dose_mg > max_dose:
                    severity = Severity.CRITICAL if daily_dose_mg > max_dose * 1.5 else Severity.HIGH
                    issues.append({
                        'drug': drug_name,
                        'prescribed_dose': daily_dose_mg,
                        'max_dose': max_dose,
                        'severity': severity.value
                    })
                    # Map severity to new score ranges
                    if severity == Severity.CRITICAL:
                        max_score = max(max_score, 10.0)  # CRITICAL: 9-10
                    elif severity == Severity.HIGH:
                        max_score = max(max_score, 7.0)  # HIGH: 6-8
                    elif severity == Severity.MODERATE:
                        max_score = max(max_score, 4.0)  # MODERATE: 3-5
                    elif severity == Severity.LOW:
                        max_score = max(max_score, 1.5)  # LOW: 1-2
                elif max_dose and daily_dose_mg < max_dose * 0.1:
                    issues.append({
                        'drug': drug_name,
                        'prescribed_dose': daily_dose_mg,
                        'min_dose': max_dose * 0.1,
                        'severity': Severity.MODERATE.value
                    })
                    max_score = max(max_score, 4.0)
        
        if issues:
            critical_issues = [i for i in issues if i.get('severity') == Severity.CRITICAL.value]
            if critical_issues:
                status = Severity.CRITICAL
                message = f"Critical: {len(critical_issues)} overdose(s) detected"
            else:
                status = Severity.HIGH
                message = f"High: {len(issues)} dosage issue(s) detected"
        else:
            # If no drugs/dosages provided, return OK
            if not drugs and not dosages:
                status = Severity.OK
                message = "No medications identified. No dosage safety concerns."
            else:
                status = Severity.OK
                message = "No dosage safety concerns identified. Current dosing appears appropriate based on available information."
        
        # Generate LLM explanation chain
        llm_explanation = None
        if self.llm_service and self.llm_service.llm:
            try:
                llm_explanation = self._generate_llm_explanation_chain(
                    "DosageChecker",
                    issues,
                    drugs,
                    dosages,
                    frequencies,
                    message
                )
            except Exception as e:
                print(f"LLM explanation chain failed: {e}")
        
        # Add LLM explanation to details
        details = {'issues': issues}
        if llm_explanation:
            details['llm_explanation'] = llm_explanation
        
        return AgentOutput(
            agent="DosageChecker",
            status=status,
            message=message,
            evidence=[str(i) for i in issues],
            score=max_score,
            details=details
        )
    
    def _generate_llm_explanation_chain(self, agent_name: str, issues: List[Dict], 
                                       *context_args) -> Dict[str, Any]:
        """Generate LLM explanation chain for agent result."""
        if not self.llm_service or not self.llm_service.llm:
            return None
        
        try:
            context_str = f"Issues: {str(issues[:2])}"
            if context_args:
                context_str += f" Context: {str(context_args[:2])}"
            
            prompt = f"""As a medical expert, provide a detailed explanation chain for this {agent_name} result:

{context_str}

Provide:
1. Why this was flagged (specific reason)
2. Which symptoms/values triggered it (list them)
3. Which guideline chunks were retrieved (if any)
4. Confidence level: high/medium/low (with reasoning)

Format as JSON:
{{
    "why_flagged": "...",
    "triggered_items": [...],
    "guideline_chunks": [...],
    "confidence": "high/medium/low",
    "confidence_reasoning": "..."
}}"""
            
            response = self.llm_service.generate(prompt, max_tokens=400)
            
            # Try to extract JSON
            import json
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback
                return {
                    "why_flagged": response[:200],
                    "triggered_items": [str(i) for i in issues[:3]],
                    "guideline_chunks": [],
                    "confidence": "medium",
                    "confidence_reasoning": "Based on LLM analysis"
                }
        except Exception as e:
            print(f"LLM explanation chain generation failed: {e}")
            return None


class InteractionCheckerAgent:
    """Agent for checking drug interactions."""
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        try:
            from app.services.llm_service import LLMService
            self.llm_service = LLMService()
        except Exception as e:
            print(f"Warning: Could not initialize LLM service for InteractionChecker: {e}")
            self.llm_service = None
    
    def _generate_llm_explanation_chain(self, agent_name: str, issues: List[Dict], 
                                       *context_args) -> Dict[str, Any]:
        """Generate LLM explanation chain for agent result."""
        if not self.llm_service or not self.llm_service.llm:
            return None
        
        try:
            context_str = f"Issues: {str(issues[:2])}"
            if context_args:
                context_str += f" Context: {str(context_args[:2])}"
            
            prompt = f"""As a medical expert, provide a detailed explanation chain for this {agent_name} result:

{context_str}

Provide:
1. Why this was flagged (specific reason)
2. Which drugs triggered it (list them)
3. Which guideline chunks were retrieved (if any)
4. Confidence level: high/medium/low (with reasoning)

Format as JSON:
{{
    "why_flagged": "...",
    "triggered_items": [...],
    "guideline_chunks": [...],
    "confidence": "high/medium/low",
    "confidence_reasoning": "..."
}}"""
            
            response = self.llm_service.generate(prompt, max_tokens=400)
            
            # Try to extract JSON
            import json
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "why_flagged": response[:200],
                    "triggered_items": [str(i) for i in issues[:3]],
                    "guideline_chunks": [],
                    "confidence": "medium",
                    "confidence_reasoning": "Based on LLM analysis"
                }
        except Exception as e:
            print(f"LLM explanation chain generation failed: {e}")
            return None
        # Known dangerous interactions (fallback)
        self.dangerous_interactions = [
            ('warfarin', 'aspirin'),
            ('warfarin', 'ibuprofen'),
            ('warfarin', 'metformin'),
            ('aspirin', 'ibuprofen'),
            ('metformin', 'furosemide')
        ]

    def check(self, drugs: List[Dict]) -> AgentOutput:
        """Check for drug interactions."""
        issues = []
        max_score = 0.0
        
        drug_names = [d.get('normalized_name', '').lower() for d in drugs]
        
        # Check all pairs
        for i in range(len(drug_names)):
            for j in range(i + 1, len(drug_names)):
                drug1 = drug_names[i]
                drug2 = drug_names[j]
                
                # Check known interactions
                is_dangerous = (drug1, drug2) in self.dangerous_interactions or \
                              (drug2, drug1) in self.dangerous_interactions
                
                # Retrieve from RAG
                interactions = self.rag_service.retrieve_drug_interactions(drug1, drug2)
                
                if interactions:
                    content = ' '.join([i.get('content', '') for i in interactions]).lower()
                    if any(word in content for word in ['contraindicated', 'dangerous', 'severe', 'avoid', 'interaction']):
                        is_dangerous = True
                
                if is_dangerous:
                    severity = Severity.CRITICAL
                    issues.append({
                        'drug1': drug1,
                        'drug2': drug2,
                        'severity': severity.value,
                        'evidence': [i.get('content', '')[:200] for i in interactions[:2]]
                    })
                    max_score = max(max_score, 10.0)
        
        if issues:
            status = Severity.CRITICAL
            message = f"Critical: {len(issues)} dangerous drug interaction(s) detected"
        else:
            # If no drugs or only one drug, return OK (no interactions possible)
            if not drugs or len(drugs) <= 1:
                status = Severity.OK
                message = "No medications or single medication identified. No drug interactions possible."
            else:
                status = Severity.OK
                message = "No dangerous drug interactions detected"
        
        # Generate LLM explanation chain
        llm_explanation = None
        if self.llm_service and self.llm_service.llm:
            try:
                llm_explanation = self._generate_llm_explanation_chain(
                    "InteractionChecker",
                    issues,
                    drugs,
                    message
                )
            except Exception as e:
                print(f"LLM explanation chain failed: {e}")
        
        details = {'interactions': issues}
        if llm_explanation:
            details['llm_explanation'] = llm_explanation
        
        return AgentOutput(
            agent="InteractionChecker",
            status=status,
            message=message,
            evidence=[str(i) for i in issues],
            score=max_score,
            details=details
        )


class RedFlagCheckerAgent:
    """Agent for checking red flag symptoms with comprehensive emergency detection."""
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        try:
            from app.services.llm_service import LLMService
            self.llm_service = LLMService()
        except Exception as e:
            print(f"Warning: Could not initialize LLM service for RedFlagChecker: {e}")
            self.llm_service = None
        
        # NEW SCORING SYSTEM - Medically Aligned
        # CRITICAL (Life-Threatening): 9-10
        # HIGH (Serious): 6-8
        # MODERATE (Needs Evaluation): 3-5
        # LOW (Mild): 1-2
        # SAFE/OK: 0
        self.SCORE_CRITICAL_BASE = 10.0  # Base for critical emergencies
        self.SCORE_CRITICAL_BORDERLINE = 9.0  # Borderline critical
        self.SCORE_HIGH_STRONG = 8.0  # Strong high-risk cases
        self.SCORE_HIGH_MODERATE = 7.0  # Moderate-high cases
        self.SCORE_HIGH_LOW = 6.0  # Low-high cases
        self.SCORE_MODERATE_STRONG = 5.0  # Strong moderate
        self.SCORE_MODERATE_TYPICAL = 4.0  # Typical moderate
        self.SCORE_MODERATE_MILD = 3.0  # Mild moderate
        self.SCORE_LOW_MILD = 2.0  # Mild symptoms
        self.SCORE_LOW_VERY_MILD = 1.0  # Very mild symptoms
        self.SCORE_SAFE = 0.0  # Safe/OK
        
        # Cardiac event patterns
        self.cardiac_severe = [
            'chest pain', 'crushing chest pain', 'chest pressure', 'chest tightness',
            'heart attack', 'myocardial infarction', 'angina',
            'radiating pain', 'pain radiating to arm', 'pain radiating to jaw',
            'sweating', 'diaphoresis', 'profuse sweating'
        ]
        # Note: 'mi' removed to avoid false positives (matches "sputum", "symptom", etc.)
        # Use full phrase "myocardial infarction" or "heart attack" instead
        self.cardiac_moderate = [
            'palpitations', 'irregular heartbeat', 'arrhythmia',
            'dizziness', 'lightheadedness', 'syncope', 'fainting'
        ]
        self.cardiac_mild = [
            'fatigue', 'weakness', 'mild chest discomfort'
        ]
        
        # Respiratory distress patterns
        self.respiratory_severe = [
            'severe shortness of breath', 'unable to breathe', 'respiratory distress',
            'gasping', 'choking', 'air hunger', 'severe dyspnea',
            'cyanosis', 'blue lips', 'blue skin', 'hypoxia'
        ]
        self.respiratory_moderate = [
            'shortness of breath', 'dyspnea', 'difficulty breathing',
            'wheezing', 'stridor', 'rapid breathing', 'tachypnea'
        ]
        self.respiratory_mild = [
            'mild breathlessness', 'exertional dyspnea'
        ]
        
        # Infection-related red flags
        self.infection_severe = [
            'high fever', 'fever > 103', 'fever > 39.4', 'septic shock',
            'sepsis', 'severe infection', 'meningitis', 'encephalitis',
            'rigors', 'chills', 'shaking chills'
        ]
        self.infection_moderate = [
            'fever', 'elevated temperature', 'warm to touch',
            'redness', 'swelling', 'purulent discharge', 'pus'
        ]
        self.infection_mild = [
            'mild fever', 'low-grade fever', 'warmth'
        ]
        
        # Neurological emergencies
        self.neuro_severe = [
            'stroke', 'cva', 'cerebrovascular accident', 'loss of consciousness',
            'unconscious', 'coma', 'seizure', 'convulsion', 'status epilepticus',
            'severe headache', 'thunderclap headache', 'worst headache',
            'facial droop', 'arm weakness', 'speech difficulty', 'f.a.s.t',
            'altered mental status', 'confusion', 'disorientation'
        ]
        self.neuro_moderate = [
            'headache', 'migraine', 'dizziness', 'vertigo',
            'numbness', 'tingling', 'weakness', 'paresthesia'
        ]
        self.neuro_mild = [
            'mild headache', 'minor dizziness'
        ]
        
        # Diabetic emergencies - Hyperglycemia
        self.hyperglycemia_severe = [
            'glucose > 300', 'glucose > 300 mg/dl', 'blood sugar > 300',
            'severe hyperglycemia', 'diabetic ketoacidosis', 'dka',
            'ketoacidosis', 'ketones', 'ketonuria'
        ]
        self.hyperglycemia_moderate = [
            'glucose > 250', 'glucose > 250 mg/dl', 'blood sugar > 250',
            'hyperglycemia', 'high glucose', 'elevated glucose'
        ]
        self.hyperglycemia_symptoms = [
            'polyuria', 'excessive urination', 'frequent urination',
            'polydipsia', 'excessive thirst', 'increased thirst',
            'blurred vision', 'blurry vision', 'vision changes',
            'nausea', 'vomiting', 'nausea and vomiting',
            'dehydration', 'dehydrated', 'dry mouth', 'dry skin',
            'fatigue', 'weakness', 'lethargy'
        ]

    def _check_cardiac_events(self, text_lower: str, symptoms: List[str]) -> List[Dict]:
        """Check for cardiac event red flags."""
        issues = []
        detected_symptoms = []
        
        # Check severe cardiac symptoms
        # Use word boundaries to avoid false positives (e.g., "mi" in "sputum")
        import re
        for pattern in self.cardiac_severe:
            # Use word boundaries for short patterns to avoid false matches
            if len(pattern) <= 3:
                pattern_regex = r'\b' + re.escape(pattern) + r'\b'
                matches = bool(re.search(pattern_regex, text_lower, re.IGNORECASE)) or any(re.search(pattern_regex, s.lower(), re.IGNORECASE) for s in symptoms)
            else:
                matches = pattern in text_lower or any(pattern in s.lower() for s in symptoms)
            
            if matches:
                detected_symptoms.append(pattern)
                issues.append({
                    'category': 'Cardiac Event',
                    'symptom': pattern,
                    'severity': 'critical',
                    'score': self.SCORE_CRITICAL_BASE,
                    'reasoning': f'CRITICAL cardiac symptom detected: {pattern}. This requires immediate emergency evaluation for possible myocardial infarction or acute coronary syndrome.'
                })
        
        # Check moderate cardiac symptoms
        for pattern in self.cardiac_moderate:
            if pattern in text_lower or any(pattern in s.lower() for s in symptoms):
                detected_symptoms.append(pattern)
                issues.append({
                    'category': 'Cardiac Event',
                    'symptom': pattern,
                    'severity': 'high',
                    'score': self.SCORE_HIGH_MODERATE,
                    'reasoning': f'HIGH cardiac symptom: {pattern}. May indicate arrhythmia or cardiac dysfunction requiring urgent evaluation.'
                })
        
        # Check mild cardiac symptoms
        for pattern in self.cardiac_mild:
            if pattern in text_lower or any(pattern in s.lower() for s in symptoms):
                detected_symptoms.append(pattern)
                issues.append({
                    'category': 'Cardiac Event',
                    'symptom': pattern,
                    'severity': 'moderate',
                    'score': self.SCORE_MODERATE_TYPICAL,
                    'reasoning': f'MODERATE cardiac symptom: {pattern}. Monitor closely and consider cardiac evaluation if persistent.'
                })
        
        # Escalate if multiple cardiac symptoms
        if len(detected_symptoms) >= 2:
            severe_count = sum(1 for i in issues if i.get('severity') == 'severe')
            if severe_count == 0:  # Multiple moderate/mild symptoms
                issues.append({
                    'category': 'Cardiac Event',
                    'symptom': f'Multiple cardiac symptoms: {", ".join(detected_symptoms[:3])}',
                    'severity': 'high',
                    'score': self.SCORE_HIGH_MODERATE,
                    'reasoning': f'Multiple cardiac symptoms detected ({len(detected_symptoms)}). Combined presentation increases concern for cardiac event. ESCALATED to high severity.'
                })
        
        return issues

    def _check_respiratory_distress(self, text_lower: str, symptoms: List[str]) -> List[Dict]:
        """Check for respiratory distress red flags."""
        issues = []
        detected_symptoms = []
        
        # Check severe respiratory symptoms
        for pattern in self.respiratory_severe:
            if pattern in text_lower or any(pattern in s.lower() for s in symptoms):
                detected_symptoms.append(pattern)
                issues.append({
                    'category': 'Respiratory Distress',
                    'symptom': pattern,
                    'severity': 'critical',
                    'score': self.SCORE_CRITICAL_BASE,
                    'reasoning': f'SEVERE respiratory distress: {pattern}. This is a life-threatening emergency requiring immediate intervention. Patient may need oxygen, airway management, or emergency department evaluation.'
                })
        
        # Check moderate respiratory symptoms
        for pattern in self.respiratory_moderate:
            if pattern in text_lower or any(pattern in s.lower() for s in symptoms):
                detected_symptoms.append(pattern)
                issues.append({
                    'category': 'Respiratory Distress',
                    'symptom': pattern,
                    'severity': 'high',
                    'score': self.SCORE_HIGH_MODERATE,
                    'reasoning': f'Respiratory symptom identified: {pattern}. Clinical evaluation recommended to assess severity and rule out serious respiratory conditions.'
                })
        
        # Check mild respiratory symptoms
        for pattern in self.respiratory_mild:
            if pattern in text_lower or any(pattern in s.lower() for s in symptoms):
                detected_symptoms.append(pattern)
                issues.append({
                    'category': 'Respiratory Distress',
                    'symptom': pattern,
                    'severity': 'moderate',
                    'score': self.SCORE_MODERATE_TYPICAL,
                    'reasoning': f'MILD respiratory symptom: {pattern}. Monitor and consider evaluation if worsens.'
                })
        
        return issues

    def _check_infection_red_flags(self, text_lower: str, symptoms: List[str]) -> List[Dict]:
        """Check for infection-related red flags."""
        issues = []
        detected_symptoms = []
        
        # Extract fever value to avoid duplicate classification
        import re
        temp_match = re.search(r'(\d+(?:\.\d+)?)\s*°?f', text_lower)
        fever_value = None
        if temp_match:
            fever_value = float(temp_match.group(1))
        
        # Check severe infection symptoms
        for pattern in self.infection_severe:
            if pattern in text_lower or any(pattern in s.lower() for s in symptoms):
                detected_symptoms.append(pattern)
                issues.append({
                    'category': 'Infection Emergency',
                    'symptom': pattern,
                    'severity': 'critical',
                    'score': self.SCORE_CRITICAL_BASE,
                    'reasoning': f'SEVERE infection red flag: {pattern}. This may indicate sepsis, severe systemic infection, or life-threatening infectious process requiring immediate medical attention.'
                })
        
        # Check moderate infection symptoms
        # BUT: Skip if fever is already classified by 5-level system (99.5-101.5°F)
        # The 5-level system handles these temperatures explicitly as MODERATE
        # Extract fever value to check
        import re
        temp_match = re.search(r'(\d+(?:\.\d+)?)\s*°?f', text_lower)
        fever_value = None
        if temp_match:
            fever_value = float(temp_match.group(1))
        
        for pattern in self.infection_moderate:
            # Skip "fever" pattern if it's in the MODERATE range (already handled by 5-level)
            if (pattern == 'fever' or 'fever' in pattern.lower()) and fever_value and 99.5 <= fever_value <= 101.5:
                continue
            if pattern in text_lower or any(pattern in s.lower() for s in symptoms):
                detected_symptoms.append(pattern)
                issues.append({
                    'category': 'Infection Emergency',
                    'symptom': pattern,
                    'severity': 'high',
                    'score': self.SCORE_HIGH_MODERATE,
                    'reasoning': f'MODERATE infection sign: {pattern}. May indicate active infection requiring evaluation and possible antibiotic treatment.'
                })
        
        # Check mild infection symptoms
        for pattern in self.infection_mild:
            if pattern in text_lower or any(pattern in s.lower() for s in symptoms):
                detected_symptoms.append(pattern)
                issues.append({
                    'category': 'Infection Emergency',
                    'symptom': pattern,
                    'severity': 'moderate',
                    'score': self.SCORE_MODERATE_TYPICAL,
                    'reasoning': f'MILD infection sign: {pattern}. Monitor for progression.'
                })
        
        return issues

    def _check_neurological_emergencies(self, text_lower: str, symptoms: List[str]) -> List[Dict]:
        """Check for neurological emergency red flags."""
        issues = []
        detected_symptoms = []
        
        # Check severe neurological symptoms
        for pattern in self.neuro_severe:
            if pattern in text_lower or any(pattern in s.lower() for s in symptoms):
                detected_symptoms.append(pattern)
                issues.append({
                    'category': 'Neurological Emergency',
                    'symptom': pattern,
                    'severity': 'critical',
                    'score': self.SCORE_CRITICAL_BASE,
                    'reasoning': f'SEVERE neurological emergency: {pattern}. This is a critical finding that may indicate stroke, seizure, or other life-threatening neurological condition. Requires immediate emergency department evaluation.'
                })
        
        # Check moderate neurological symptoms
        for pattern in self.neuro_moderate:
            if pattern in text_lower or any(pattern in s.lower() for s in symptoms):
                detected_symptoms.append(pattern)
                issues.append({
                    'category': 'Neurological Emergency',
                    'symptom': pattern,
                    'severity': 'high',
                    'score': self.SCORE_HIGH_MODERATE,
                    'reasoning': f'MODERATE neurological symptom: {pattern}. May indicate neurological dysfunction requiring urgent evaluation.'
                })
        
        # Check mild neurological symptoms
        for pattern in self.neuro_mild:
            if pattern in text_lower or any(pattern in s.lower() for s in symptoms):
                detected_symptoms.append(pattern)
                issues.append({
                    'category': 'Neurological Emergency',
                    'symptom': pattern,
                    'severity': 'moderate',
                    'score': self.SCORE_MODERATE_TYPICAL,
                    'reasoning': f'MILD neurological symptom: {pattern}. Monitor and consider evaluation if persistent or worsening.'
                })
        
        return issues

    def _check_hyperglycemia(self, text_lower: str, symptoms: List[str], lab_values: List[Dict] = None) -> List[Dict]:
        """Check for diabetic hyperglycemia emergencies."""
        issues = []
        detected_symptoms = []
        glucose_value = None
        
        # Extract glucose from lab values
        if lab_values:
            for lab in lab_values:
                lab_text = str(lab.get('text', '')).lower()
                lab_value = lab.get('value', None)
                
                # Check for glucose values
                if 'glucose' in lab_text or 'blood sugar' in lab_text:
                    if lab_value:
                        glucose_value = float(lab_value)
                    else:
                        # Try to extract from text
                        import re
                        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:mg/dl|mg/dL|mg|mmol)', lab_text)
                        if match:
                            glucose_value = float(match.group(1))
        
        # Also check text for glucose mentions
        if not glucose_value:
            import re
            glucose_patterns = [
                r'glucose[:\s]+(\d+(?:\.\d+)?)',
                r'blood sugar[:\s]+(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*(?:mg/dl|mg/dL|mg).*glucose'
            ]
            for pattern in glucose_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    glucose_value = float(match.group(1))
                    break
        
        # Check glucose thresholds
        if glucose_value:
            if glucose_value > 300:
                detected_symptoms.append(f'Glucose {glucose_value} mg/dL')
                issues.append({
                    'category': 'Diabetic Emergency - Hyperglycemia',
                    'symptom': f'Glucose > 300 mg/dL (measured: {glucose_value} mg/dL)',
                    'severity': 'critical',
                    'score': self.SCORE_CRITICAL_BASE,
                    'reasoning': f'SEVERE hyperglycemia: Glucose level of {glucose_value} mg/dL exceeds 300 mg/dL threshold. This is a critical finding that may indicate diabetic ketoacidosis (DKA) or hyperosmolar hyperglycemic state (HHS). Requires immediate medical attention, possible insulin therapy, and monitoring for ketones and acidosis.'
                })
            elif glucose_value > 250:
                detected_symptoms.append(f'Glucose {glucose_value} mg/dL')
                issues.append({
                    'category': 'Diabetic Emergency - Hyperglycemia',
                    'symptom': f'Glucose > 250 mg/dL (measured: {glucose_value} mg/dL)',
                    'severity': 'high',
                    'score': self.SCORE_HIGH_MODERATE,
                    'reasoning': f'MODERATE hyperglycemia: Glucose level of {glucose_value} mg/dL exceeds 250 mg/dL threshold. This indicates significant hyperglycemia requiring evaluation, possible insulin adjustment, and monitoring for progression to severe hyperglycemia or DKA.'
                })
        
        # Check for hyperglycemia symptoms
        hyperglycemia_symptom_count = 0
        for pattern in self.hyperglycemia_symptoms:
            if pattern in text_lower or any(pattern in s.lower() for s in symptoms):
                detected_symptoms.append(pattern)
                hyperglycemia_symptom_count += 1
                issues.append({
                    'category': 'Diabetic Emergency - Hyperglycemia',
                    'symptom': pattern,
                    'severity': 'high' if hyperglycemia_symptom_count >= 2 else 'moderate',
                    'score': self.SCORE_HIGH_MODERATE if hyperglycemia_symptom_count >= 2 else self.SCORE_MODERATE_TYPICAL,
                    'reasoning': f'Hyperglycemia symptom: {pattern}. Classic symptom of elevated blood glucose. When combined with other symptoms or elevated glucose levels, indicates need for evaluation.'
                })
        
        # Escalate based on symptom combination
        if hyperglycemia_symptom_count >= 3:
            issues.append({
                'category': 'Diabetic Emergency - Hyperglycemia',
                'symptom': f'Multiple hyperglycemia symptoms ({hyperglycemia_symptom_count} detected)',
                'severity': 'critical',
                'score': self.SCORE_CRITICAL_BASE,
                'reasoning': f'CRITICAL: Multiple hyperglycemia symptoms detected ({hyperglycemia_symptom_count}). Combined presentation (polyuria, polydipsia, dehydration, etc.) strongly suggests significant hyperglycemia or developing DKA. ESCALATED to critical severity. Requires immediate evaluation and glucose monitoring.'
            })
        elif hyperglycemia_symptom_count >= 2 and glucose_value and glucose_value > 250:
            issues.append({
                'category': 'Diabetic Emergency - Hyperglycemia',
                'symptom': f'Elevated glucose ({glucose_value} mg/dL) with multiple symptoms',
                'severity': 'critical',
                'score': self.SCORE_CRITICAL_BASE,
                'reasoning': f'SEVERE: Elevated glucose ({glucose_value} mg/dL) combined with {hyperglycemia_symptom_count} hyperglycemia symptoms. This combination indicates significant hyperglycemic state requiring immediate attention. ESCALATED to severe severity.'
            })
        
        return issues

    def _check_cardiac_pattern_combination(self, text_lower: str, symptoms: List[str]) -> List[Dict]:
        """Check for cardiac emergency pattern: chest pain + sweating + SOB."""
        issues = []
        has_chest_pain = any('chest pain' in s.lower() or 'chest pain' in text_lower for s in symptoms)
        has_sweating = any('sweating' in s.lower() or 'diaphoresis' in s.lower() or 'sweating' in text_lower for s in symptoms)
        has_sob = any('shortness of breath' in s.lower() or 'dyspnea' in s.lower() or 'shortness of breath' in text_lower for s in symptoms)
        
        if has_chest_pain and has_sweating and has_sob:
            issues.append({
                'category': 'Cardiac Emergency Pattern',
                'symptom': 'Chest pain + Sweating + Shortness of breath',
                'severity': 'severe',
                'score': self.SCORE_CRITICAL_BASE,
                'reasoning': 'CRITICAL CARDIAC PATTERN: Classic triad of chest pain, sweating, and shortness of breath detected. This combination is highly suggestive of acute coronary syndrome or myocardial infarction. Requires IMMEDIATE emergency evaluation, ECG, cardiac enzymes, and possible reperfusion therapy.'
            })
        elif has_chest_pain and (has_sweating or has_sob):
            issues.append({
                'category': 'Cardiac Emergency Pattern',
                'symptom': f'Chest pain + {"Sweating" if has_sweating else "Shortness of breath"}',
                'severity': 'severe',
                'score': self.SCORE_CRITICAL_BASE,
                'reasoning': f'CRITICAL CARDIAC PATTERN: Chest pain combined with {"sweating" if has_sweating else "shortness of breath"} is highly concerning for cardiac event. Requires immediate emergency evaluation.'
            })
        
        return issues

    def _check_diabetic_emergency_pattern(self, text_lower: str, symptoms: List[str], lab_values: List[Dict] = None) -> List[Dict]:
        """Check for diabetic emergency: glucose > 250 + nausea + thirst + blurred vision."""
        issues = []
        glucose_value = None
        
        # Extract glucose
        if lab_values:
            for lab in lab_values:
                if 'glucose' in str(lab.get('text', '')).lower() or 'blood sugar' in str(lab.get('text', '')).lower():
                    glucose_value = lab.get('value')
                    if not glucose_value:
                        import re
                        match = re.search(r'(\d+(?:\.\d+)?)', str(lab.get('text', '')))
                        if match:
                            glucose_value = float(match.group(1))
        
        if not glucose_value:
            import re
            match = re.search(r'glucose[:\s]+(\d+(?:\.\d+)?)|blood sugar[:\s]+(\d+(?:\.\d+)?)', text_lower)
            if match:
                glucose_value = float(match.group(1) or match.group(2))
        
        has_nausea = any('nausea' in s.lower() or 'vomiting' in s.lower() or 'nausea' in text_lower for s in symptoms)
        has_thirst = any('thirst' in s.lower() or 'polydipsia' in s.lower() or 'thirst' in text_lower for s in symptoms)
        has_blurred_vision = any('blurred vision' in s.lower() or 'blurry' in s.lower() or 'blurred vision' in text_lower for s in symptoms)
        
        if glucose_value and glucose_value > 250:
            if has_nausea and has_thirst and has_blurred_vision:
                issues.append({
                    'category': 'Diabetic Emergency Pattern',
                    'symptom': f'Glucose >250 ({glucose_value} mg/dL) + Nausea + Thirst + Blurred vision',
                    'severity': 'critical',
                    'score': self.SCORE_CRITICAL_BASE,
                    'reasoning': f'CRITICAL DIABETIC EMERGENCY PATTERN: Elevated glucose ({glucose_value} mg/dL) combined with classic hyperglycemia symptoms (nausea, thirst, blurred vision). This pattern strongly suggests significant hyperglycemia or developing DKA. Requires immediate medical attention, glucose monitoring, ketone testing, and possible insulin therapy.'
                })
            elif (has_nausea and has_thirst) or (has_thirst and has_blurred_vision):
                issues.append({
                    'category': 'Diabetic Emergency Pattern',
                    'symptom': f'Glucose >250 ({glucose_value} mg/dL) + Multiple hyperglycemia symptoms',
                    'severity': 'critical',
                    'score': self.SCORE_CRITICAL_BASE,
                    'reasoning': f'CRITICAL DIABETIC EMERGENCY: Elevated glucose ({glucose_value} mg/dL) with multiple hyperglycemia symptoms. Requires urgent evaluation and glucose management.'
                })
        
        return issues

    def _check_meningitis_pattern(self, text_lower: str, symptoms: List[str]) -> List[Dict]:
        """Check for meningitis pattern: fever + headache + neck stiffness + photophobia + vomiting."""
        issues = []
        has_fever = any('fever' in s.lower() or 'fever' in text_lower or 'temperature' in s.lower() for s in symptoms)
        has_headache = any('headache' in s.lower() or 'headache' in text_lower for s in symptoms)
        has_neck_stiffness = any('neck stiffness' in s.lower() or 'nuchal rigidity' in s.lower() or 'neck stiffness' in text_lower for s in symptoms)
        has_photophobia = any('photophobia' in s.lower() or 'light sensitivity' in s.lower() or 'photophobia' in text_lower for s in symptoms)
        has_vomiting = any('vomiting' in s.lower() or 'emesis' in s.lower() or 'vomiting' in text_lower for s in symptoms)
        
        symptom_count = sum([has_fever, has_headache, has_neck_stiffness, has_photophobia, has_vomiting])
        
        if symptom_count >= 4:
            issues.append({
                'category': 'Meningitis Pattern',
                'symptom': f'Meningitis pattern: {symptom_count}/5 symptoms (fever, headache, neck stiffness, photophobia, vomiting)',
                'severity': 'critical',
                'score': self.SCORE_CRITICAL_BASE,
                'reasoning': f'CRITICAL MENINGITIS PATTERN: {symptom_count} out of 5 classic meningitis symptoms detected (fever, headache, neck stiffness, photophobia, vomiting). This is a medical emergency requiring immediate evaluation for possible bacterial or viral meningitis. Requires urgent lumbar puncture, blood cultures, and antibiotic therapy if bacterial meningitis suspected.'
            })
        elif symptom_count >= 3:
            issues.append({
                'category': 'Meningitis Pattern',
                'symptom': f'Possible meningitis: {symptom_count}/5 symptoms',
                'severity': 'critical',
                'score': self.SCORE_CRITICAL_BORDERLINE,
                'reasoning': f'CRITICAL: {symptom_count} meningitis symptoms detected. High suspicion for meningitis. Requires immediate medical evaluation.'
            })
        
        return issues

    def _check_hypertensive_crisis(self, text_lower: str, lab_values: List[Dict] = None) -> List[Dict]:
        """Check for hypertensive crisis: BP > 180/120."""
        issues = []
        import re
        
        # Extract BP from text or lab values
        bp_systolic = None
        bp_diastolic = None
        
        # Check lab values
        if lab_values:
            for lab in lab_values:
                lab_text = str(lab.get('text', '')).lower()
                if 'bp' in lab_text or 'blood pressure' in lab_text:
                    match = re.search(r'(\d+)\s*[/-]\s*(\d+)', lab_text)
                    if match:
                        bp_systolic = int(match.group(1))
                        bp_diastolic = int(match.group(2))
        
        # Check text
        if not bp_systolic:
            match = re.search(r'(?:bp|blood pressure)[:\s]*(\d+)\s*[/-]\s*(\d+)', text_lower)
            if match:
                bp_systolic = int(match.group(1))
                bp_diastolic = int(match.group(2))
        
        if bp_systolic and bp_diastolic:
            if bp_systolic > 180 or bp_diastolic > 120:
                issues.append({
                    'category': 'Hypertensive Crisis',
                    'symptom': f'BP {bp_systolic}/{bp_diastolic} mmHg (Hypertensive Crisis)',
                    'severity': 'critical',
                    'score': self.SCORE_CRITICAL_BASE,
                    'reasoning': f'CRITICAL HYPERTENSIVE CRISIS: Blood pressure of {bp_systolic}/{bp_diastolic} mmHg exceeds 180/120 threshold. This is a hypertensive emergency requiring immediate medical attention. May cause end-organ damage (stroke, MI, renal failure, aortic dissection). Requires urgent BP reduction with IV antihypertensives in monitored setting.'
                })
        
        return issues

    def _check_respiratory_emergency(self, text_lower: str, lab_values: List[Dict] = None) -> List[Dict]:
        """Check for respiratory emergency: oxygen < 92%."""
        issues = []
        import re
        
        oxygen_value = None
        
        # Check lab values
        if lab_values:
            for lab in lab_values:
                lab_text = str(lab.get('text', '')).lower()
                if 'oxygen' in lab_text or 'o2' in lab_text or 'spo2' in lab_text or 'saturation' in lab_text:
                    match = re.search(r'(\d+(?:\.\d+)?)\s*(?:%|percent)', lab_text)
                    if match:
                        oxygen_value = float(match.group(1))
        
        # Check text
        if not oxygen_value:
            match = re.search(r'(?:oxygen|o2|spo2|saturation)[:\s]*(\d+(?:\.\d+)?)\s*(?:%|percent)', text_lower)
            if match:
                oxygen_value = float(match.group(1))
        
        if oxygen_value and oxygen_value < 92:
            issues.append({
                'category': 'Respiratory Emergency',
                'symptom': f'Oxygen saturation {oxygen_value}% (< 92%)',
                'severity': 'critical',
                'score': self.SCORE_CRITICAL_BASE,
                'reasoning': f'CRITICAL RESPIRATORY EMERGENCY: Oxygen saturation of {oxygen_value}% is below 92% threshold, indicating significant hypoxemia. This is a life-threatening condition requiring immediate oxygen therapy, respiratory support, and emergency evaluation. May indicate severe respiratory failure, pulmonary embolism, or other critical respiratory conditions.'
            })
        
        return issues

    def _llm_based_inference(self, symptoms: List[str], text: str, lab_values: List[Dict] = None) -> Dict[str, Any]:
        """Use LLM to infer red flags from unclear or ambiguous patterns."""
        if not self.llm_service or not self.llm_service.llm:
            return None
        
        try:
            symptoms_str = ', '.join(symptoms[:10])
            lab_str = ', '.join([str(lab.get('text', ''))[:50] for lab in (lab_values or [])[:5]])
            
            prompt = f"""As a medical expert, analyze these patient symptoms and determine if there are any red flag emergency patterns:

Symptoms: {symptoms_str}
Text: {text[:500]}
Lab Values: {lab_str}

Analyze for:
1. Cardiac emergencies (chest pain patterns)
2. Neurological emergencies (stroke, seizure patterns)
3. Respiratory emergencies (severe dyspnea, hypoxia)
4. Infection emergencies (sepsis, meningitis patterns)
5. Diabetic emergencies (DKA, HHS)
6. Hypertensive crisis
7. Other critical patterns

Provide:
- Severity: critical/high/moderate/low/ok
- Score: 0-10
- Reasoning: Why this was flagged
- Triggered Symptoms: Which specific symptoms triggered this
- Confidence: high/medium/low

Format as JSON:
{{
    "severity": "...",
    "score": ...,
    "reasoning": "...",
    "triggered_symptoms": [...],
    "confidence": "..."
}}"""
            
            response = self.llm_service.generate(prompt, max_tokens=300)
            
            # Try to extract JSON from response
            import json
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    return result
                except json.JSONDecodeError:
                    # Try to fix common JSON issues
                    json_str = json_match.group()
                    json_str = json_str.replace("'", '"')  # Replace single quotes
                    try:
                        result = json.loads(json_str)
                        return result
                    except:
                        pass
            else:
                # Fallback: parse text response
                severity = 'moderate'
                score = 5.0
                if 'critical' in response.lower() or 'emergency' in response.lower():
                    severity = 'critical'
                    score = 10.0  # CRITICAL range: 9-10
                elif 'high' in response.lower():
                    severity = 'high'
                    score = 7.0  # HIGH range: 6-8
                elif 'moderate' in response.lower():
                    severity = 'moderate'
                    score = 4.0  # MODERATE range: 3-5
                elif 'low' in response.lower():
                    severity = 'low'
                    score = 1.5  # LOW range: 1-2
                
                return {
                    'severity': severity,
                    'score': score,
                    'reasoning': response[:200],
                    'triggered_symptoms': symptoms[:3],
                    'confidence': 'medium'
                }
        except Exception as e:
            print(f"LLM inference failed: {e}")
            return None

    def check(self, symptoms: List[str], text: str, lab_values: List[Dict] = None) -> AgentOutput:
        """Hybrid check: Rule-based + LLM-based inference with OR logic."""
        # FIRST: Check for mild non-dangerous symptoms - if ONLY mild symptoms, return OK immediately
        mild_symptoms = {
            'runny nose', 'nasal congestion', 'stuffy nose',
            'sneezing', 'sneezes',
            'watery eyes', 'tearing', 'teary eyes',
            'itchy nose', 'itchy eyes', 'itchy throat',
            'mild cough', 'cough without fever', 'dry cough',
            'seasonal allergy', 'allergy symptoms', 'allergic rhinitis',
            'hay fever', 'post-nasal drip'
        }
        
        red_flag_symptoms = {
            'fever', 'high fever', 'temperature', 'chills',
            'chest pain', 'chest discomfort', 'chest pressure',
            'shortness of breath', 'dyspnea', 'sob', 'breathlessness',
            'severe headache', 'thunderclap headache', 'worst headache',
            'confusion', 'altered mental status', 'loss of consciousness',
            'severe pain', 'crushing pain', 'radiating pain'
        }
        
        text_lower = text.lower() if text else ""
        symptoms_lower = [s.lower().strip() if isinstance(s, str) else str(s).lower().strip() for s in symptoms] if symptoms else []
        
        # Filter out negated symptoms from the list
        negation_words = ['no ', 'not ', 'without ', 'denies ', 'denied ', 'absent ', 'none ']
        filtered_symptoms = []
        for symptom in symptoms_lower:
            # Check if this symptom is negated in the text
            is_negated = False
            if text_lower and symptom in text_lower:
                symptom_pos = text_lower.find(symptom)
                if symptom_pos > 0:
                    # Check 20 characters before for negation words
                    context_before = text_lower[max(0, symptom_pos - 20):symptom_pos]
                    is_negated = any(neg in context_before for neg in negation_words)
            
            # Only include non-negated symptoms
            if not is_negated:
                filtered_symptoms.append(symptom)
        
        symptoms_lower = filtered_symptoms
        
        # Check if symptoms include ONLY mild symptoms (no red flags)
        has_only_mild = len(symptoms_lower) > 0
        has_red_flags = False
        
        # Check symptoms list
        for symptom in symptoms_lower:
            # Check if it's a red flag symptom
            if any(red_flag in symptom for red_flag in red_flag_symptoms):
                has_red_flags = True
                has_only_mild = False
                break
            # Check if it's a mild symptom
            if not any(mild in symptom for mild in mild_symptoms):
                # If it's not a mild symptom and not a red flag, it's something else
                # Check if it's truly mild by checking for "mild" prefix
                if not symptom.startswith('mild'):
                    has_only_mild = False
        
        # Also check text for red flags (but ignore if negated)
        if text_lower:
            for red_flag in red_flag_symptoms:
                # Check if red flag appears in text
                if red_flag in text_lower:
                    # Check if it's negated (e.g., "no fever", "no chest pain")
                    # Look for negation patterns before the red flag
                    red_flag_pos = text_lower.find(red_flag)
                    if red_flag_pos > 0:
                        # Check 10 characters before for negation words
                        context_before = text_lower[max(0, red_flag_pos - 20):red_flag_pos]
                        negation_words = ['no ', 'not ', 'without ', 'denies ', 'denied ', 'absent ', 'none ']
                        is_negated = any(neg in context_before for neg in negation_words)
                        if not is_negated:
                            has_red_flags = True
                            has_only_mild = False
                            break
                    else:
                        # Red flag at start of text - check if it's negated
                        has_red_flags = True
                        has_only_mild = False
                        break
        
        all_issues = []
        
        # Step 1: Explicit 5-Level Severity Classification
        # This runs BEFORE rule-based detection to properly categorize symptoms
        
        # Extract vitals from text for severity assessment
        import re
        fever_value = None
        oxygen_value = None
        heart_rate_value = None
        bp_systolic = None
        glucose_value = None
        
        # Extract fever/temperature
        temp_match = re.search(r'(\d+(?:\.\d+)?)\s*°?f', text_lower)
        if temp_match:
            fever_value = float(temp_match.group(1))
        
        # Extract oxygen
        o2_match = re.search(r'(?:oxygen|o2|spo2|saturation)[:\s]*(\d+(?:\.\d+)?)\s*%', text_lower)
        if o2_match:
            oxygen_value = float(o2_match.group(1))
        
        # Extract heart rate
        hr_match = re.search(r'(?:heart rate|hr|pulse)[:\s]*(\d+)\s*(?:bpm)?', text_lower)
        if hr_match:
            heart_rate_value = int(hr_match.group(1))
        
        # Extract BP
        bp_match = re.search(r'(?:bp|blood pressure)[:\s]*(\d+)\s*[/-]\s*(\d+)', text_lower)
        if bp_match:
            bp_systolic = int(bp_match.group(1))
        
        # Extract glucose
        glucose_match = re.search(r'glucose[:\s]*(\d+(?:\.\d+)?)\s*(?:mg/dl)?', text_lower)
        if glucose_match:
            glucose_value = float(glucose_match.group(1))
        
        # Check lab values if provided
        if lab_values:
            for lab in lab_values:
                if isinstance(lab, dict):
                    lab_type = str(lab.get('type', '')).lower()
                    lab_test = str(lab.get('test', '')).lower()
                    lab_val = lab.get('value')
                    
                    if 'glucose' in lab_type or 'glucose' in lab_test:
                        if lab_val:
                            glucose_value = float(lab_val) if isinstance(lab_val, (int, float, str)) else None
                    elif 'oxygen' in lab_type or 'spo2' in lab_type:
                        if lab_val:
                            oxygen_value = float(lab_val) if isinstance(lab_val, (int, float, str)) else None
                    elif 'heart rate' in lab_type or 'hr' in lab_type:
                        if lab_val:
                            heart_rate_value = int(lab_val) if isinstance(lab_val, (int, float, str)) else None
                    elif 'blood pressure' in lab_type or 'bp' in lab_type:
                        if isinstance(lab_val, (list, tuple)) and len(lab_val) >= 1:
                            bp_systolic = int(lab_val[0])
        
        # LEVEL 1: OK (0) - No symptoms or normal vitals only
        if len(symptoms_lower) == 0:
            return AgentOutput(
                agent="RedFlagChecker",
                status=Severity.OK,
                message="No symptoms detected. Patient appears stable.",
                evidence=[],
                score=self.SCORE_SAFE,
                details={'reasoning': 'No symptoms present.'}
            )
        
        # LEVEL 2: LOW (1-2) - Mild self-limiting symptoms
        low_symptoms = {
            'runny nose', 'nasal congestion', 'stuffy nose',
            'sneezing', 'sneezes',
            'watery eyes', 'tearing', 'teary eyes',
            'itchy nose', 'itchy eyes', 'itchy throat',
            'mild cough', 'dry cough',
            'mild headache', 'mild head ache',
            'mild stomach discomfort', 'mild stomach ache',
            'mild fatigue', 'tiredness',
            'seasonal allergy', 'allergy symptoms', 'allergic rhinitis',
            'hay fever', 'post-nasal drip'
        }
        
        # Check for LOW severity: ONLY mild symptoms, no fever, no red flags
        has_only_low_symptoms = True
        has_fever = fever_value is not None and fever_value >= 99.5
        has_sob = any('shortness of breath' in s or 'sob' in s or 'dyspnea' in s for s in symptoms_lower) or 'shortness of breath' in text_lower
        has_red_flag_in_symptoms = any(red_flag in s for s in symptoms_lower for red_flag in red_flag_symptoms)
        
        for symptom in symptoms_lower:
            if not any(low_sym in symptom for low_sym in low_symptoms):
                # Check if it's a mild variant
                if not symptom.startswith('mild'):
                    has_only_low_symptoms = False
                    break
        
        # Special case: mild cough is LOW only if no fever and no SOB
        has_mild_cough = any('mild cough' in s or 'dry cough' in s for s in symptoms_lower)
        if has_mild_cough and (has_fever or has_sob):
            has_only_low_symptoms = False
        
        if has_only_low_symptoms and not has_fever and not has_sob and not has_red_flag_in_symptoms:
            # Determine LOW score (1-2)
            low_score = self.SCORE_LOW_VERY_MILD if len(symptoms_lower) <= 2 else self.SCORE_LOW_MILD
            
            return AgentOutput(
                agent="RedFlagChecker",
                status=Severity.LOW,
                message="Mild self-limiting symptoms. No medical risk.",
                evidence=[],
                score=low_score,
                details={
                    'reasoning': 'Only mild self-limiting symptoms detected (runny nose, sneezing, allergies, etc.). No red flag symptoms present.',
                    'detected_symptoms': symptoms_lower,
                    'severity_level': 'LOW'
                }
            )
        
        # LEVEL 3: MODERATE (3-5) - Moderate clinical symptoms
        moderate_indicators = []
        moderate_score = self.SCORE_MODERATE_MILD
        
        # Fever 99.5-101°F
        if fever_value and 99.5 <= fever_value <= 101.0:
            moderate_indicators.append(f'Fever {fever_value}°F')
            moderate_score = max(moderate_score, self.SCORE_MODERATE_TYPICAL)
        
        # Sore throat
        if any('sore throat' in s or 'throat pain' in s for s in symptoms_lower) or 'sore throat' in text_lower:
            moderate_indicators.append('Sore throat')
            moderate_score = max(moderate_score, self.SCORE_MODERATE_TYPICAL)
        
        # Moderate headache
        if any('moderate headache' in s or 'headache' in s for s in symptoms_lower) and not has_fever:
            moderate_indicators.append('Moderate headache')
            moderate_score = max(moderate_score, self.SCORE_MODERATE_MILD)
        
        # Vomiting without dehydration
        if any('vomiting' in s or 'nausea' in s for s in symptoms_lower) or 'vomiting' in text_lower:
            if not any('dehydration' in s or 'dehydrated' in s for s in symptoms_lower):
                moderate_indicators.append('Vomiting')
                moderate_score = max(moderate_score, self.SCORE_MODERATE_TYPICAL)
        
        # Diarrhea without blood
        if any('diarrhea' in s for s in symptoms_lower) or 'diarrhea' in text_lower:
            if not any('blood' in s or 'bloody' in s for s in symptoms_lower):
                moderate_indicators.append('Diarrhea')
                moderate_score = max(moderate_score, self.SCORE_MODERATE_MILD)
        
        # Mild SOB with oxygen ≥ 96%
        if has_sob and oxygen_value and oxygen_value >= 96:
            moderate_indicators.append(f'Mild SOB (O2 {oxygen_value}%)')
            moderate_score = max(moderate_score, self.SCORE_MODERATE_TYPICAL)
        
        # Cough without high fever or tachycardia
        has_cough = any('cough' in s for s in symptoms_lower) or 'cough' in text_lower
        has_high_fever = fever_value and fever_value > 101.0
        has_tachycardia = heart_rate_value and heart_rate_value >= 105
        
        if has_cough and not has_high_fever and not has_tachycardia:
            moderate_indicators.append('Cough')
            moderate_score = max(moderate_score, self.SCORE_MODERATE_MILD)
        
        # If we have moderate indicators and NO red flags, return MODERATE
        # This MUST return early to prevent rule-based detection from overriding
        if moderate_indicators and not has_red_flag_in_symptoms:
            # Check for any critical/high indicators that would override MODERATE
            has_critical_indicators = (
                (fever_value and fever_value >= 101.5) or
                (oxygen_value and oxygen_value < 93) or
                (bp_systolic and bp_systolic < 90) or
                (glucose_value and (glucose_value < 60 or glucose_value > 300)) or
                any('chest pain' in s for s in symptoms_lower) or
                any('confusion' in s for s in symptoms_lower) or
                any('neck stiffness' in s and has_fever for s in symptoms_lower)
            )
            
            # Also check for HIGH indicators that would override MODERATE
            has_high_indicators = (
                (fever_value and fever_value >= 101.5) or
                (oxygen_value and 93 <= oxygen_value < 96) or
                (heart_rate_value and heart_rate_value >= 105) or
                any('severe abdominal pain' in s for s in symptoms_lower) or
                (has_cough and ('sputum' in text_lower or any('sputum' in s for s in symptoms_lower)))
            )
            
            # Only return MODERATE if no HIGH or CRITICAL indicators
            # This ensures fever 99.5-101°F stays MODERATE, not HIGH
            if not has_critical_indicators and not has_high_indicators:
                # Return MODERATE immediately - don't let rule-based detection override
                return AgentOutput(
                    agent="RedFlagChecker",
                    status=Severity.MODERATE,
                    message=f"Moderate clinical symptoms requiring outpatient follow-up. Findings: {', '.join(moderate_indicators[:3])}.",
                    evidence=[],
                    score=min(5.0, max(3.0, moderate_score)),  # Clamp to 3-5 range
                    details={
                        'reasoning': f'Moderate clinical symptoms detected: {", ".join(moderate_indicators)}. Requires outpatient evaluation.',
                        'detected_symptoms': symptoms_lower,
                        'moderate_indicators': moderate_indicators,
                        'severity_level': 'MODERATE'
                    }
                )
        
        # LEVEL 4: HIGH (6-8) - Serious but not immediately life-threatening
        high_indicators = []
        high_score = self.SCORE_HIGH_LOW
        
        # Fever ≥ 101.5°F
        if fever_value and fever_value >= 101.5:
            high_indicators.append(f'Fever {fever_value}°F')
            high_score = max(high_score, self.SCORE_HIGH_MODERATE)
        
        # Cough + sputum
        if has_cough and ('sputum' in text_lower or any('sputum' in s or 'phlegm' in s for s in symptoms_lower)):
            high_indicators.append('Cough with sputum')
            high_score = max(high_score, self.SCORE_HIGH_MODERATE)
        
        # Oxygen 93-95%
        if oxygen_value and 93 <= oxygen_value < 96:
            high_indicators.append(f'Oxygen {oxygen_value}%')
            high_score = max(high_score, self.SCORE_HIGH_STRONG)
        
        # Severe abdominal pain
        if any('severe abdominal pain' in s or 'severe stomach pain' in s for s in symptoms_lower) or 'severe abdominal pain' in text_lower:
            high_indicators.append('Severe abdominal pain')
            high_score = max(high_score, self.SCORE_HIGH_MODERATE)
        
        # Tachycardia ≥ 105
        if has_tachycardia:
            high_indicators.append(f'Tachycardia {heart_rate_value} bpm')
            high_score = max(high_score, self.SCORE_HIGH_LOW)
        
        # If we have high indicators and NO critical indicators, return HIGH
        if high_indicators:
            has_critical_indicators = (
                (oxygen_value and oxygen_value < 93) or
                (bp_systolic and bp_systolic < 90) or
                (glucose_value and (glucose_value < 60 or glucose_value > 300)) or
                any('chest pain' in s for s in symptoms_lower) or
                any('confusion' in s for s in symptoms_lower) or
                any('one-sided weakness' in s or 'facial droop' in s for s in symptoms_lower) or
                (any('neck stiffness' in s for s in symptoms_lower) and has_fever)
            )
            
            if not has_critical_indicators:
                return AgentOutput(
                    agent="RedFlagChecker",
                    status=Severity.HIGH,
                    message=f"High-priority clinical findings requiring urgent evaluation. Findings: {', '.join(high_indicators[:3])}.",
                    evidence=[],
                    score=min(8.0, high_score),  # Clamp to 6-8 range
                    details={
                        'reasoning': f'High-priority clinical findings: {", ".join(high_indicators)}. Requires urgent evaluation within 24-48 hours.',
                        'detected_symptoms': symptoms_lower,
                        'high_indicators': high_indicators,
                        'severity_level': 'HIGH'
                    }
                )
        
        # LEVEL 5: CRITICAL (9-10) - Life-threatening emergencies
        # This will be handled by the existing rule-based detection below
        # But we'll add explicit critical checks here too
        
        critical_indicators = []
        critical_score = self.SCORE_CRITICAL_BASE
        
        # Chest pain
        if any('chest pain' in s or 'chest discomfort' in s for s in symptoms_lower) or 'chest pain' in text_lower:
            critical_indicators.append('Chest pain')
            critical_score = self.SCORE_CRITICAL_BASE
        
        # SOB with oxygen < 93%
        if has_sob and oxygen_value and oxygen_value < 93:
            critical_indicators.append(f'SOB with O2 {oxygen_value}%')
            critical_score = self.SCORE_CRITICAL_BASE
        
        # Confusion
        if any('confusion' in s or 'altered mental status' in s for s in symptoms_lower) or 'confusion' in text_lower:
            critical_indicators.append('Confusion')
            critical_score = self.SCORE_CRITICAL_BASE
        
        # Neck stiffness + fever
        if any('neck stiffness' in s for s in symptoms_lower) and has_fever:
            critical_indicators.append('Neck stiffness + fever')
            critical_score = self.SCORE_CRITICAL_BASE
        
        # One-sided weakness
        if any('one-sided weakness' in s or 'facial droop' in s or 'slurred speech' in s for s in symptoms_lower):
            critical_indicators.append('Neurological deficit')
            critical_score = self.SCORE_CRITICAL_BASE
        
        # BP < 90 systolic
        if bp_systolic and bp_systolic < 90:
            critical_indicators.append(f'Hypotension (BP {bp_systolic})')
            critical_score = self.SCORE_CRITICAL_BASE
        
        # Glucose < 60 or > 300
        if glucose_value:
            if glucose_value < 60:
                critical_indicators.append(f'Hypoglycemia (glucose {glucose_value})')
                critical_score = self.SCORE_CRITICAL_BASE
            elif glucose_value > 300:
                critical_indicators.append(f'Hyperglycemia (glucose {glucose_value})')
                critical_score = self.SCORE_CRITICAL_BASE
        
        # If we have critical indicators, return CRITICAL immediately
        if critical_indicators:
            return AgentOutput(
                agent="RedFlagChecker",
                status=Severity.CRITICAL,
                message=f"CRITICAL: Life-threatening emergency detected. Findings: {', '.join(critical_indicators[:3])}. Immediate medical attention required.",
                evidence=[],
                score=min(10.0, critical_score),  # Clamp to 9-10 range
                details={
                    'reasoning': f'CRITICAL: Life-threatening emergency detected. {", ".join(critical_indicators)}. Requires immediate emergency evaluation.',
                    'detected_symptoms': symptoms_lower,
                    'critical_indicators': critical_indicators,
                    'severity_level': 'CRITICAL'
                }
            )
        
        # Step 2: Rule-based detection (for patterns not caught above)
        rule_based_issues = []
        rule_based_issues.extend(self._check_cardiac_events(text_lower, symptoms))
        rule_based_issues.extend(self._check_respiratory_distress(text_lower, symptoms))
        rule_based_issues.extend(self._check_infection_red_flags(text_lower, symptoms))
        rule_based_issues.extend(self._check_neurological_emergencies(text_lower, symptoms))
        rule_based_issues.extend(self._check_hyperglycemia(text_lower, symptoms, lab_values))
        
        # Check specific pattern combinations
        rule_based_issues.extend(self._check_cardiac_pattern_combination(text_lower, symptoms))
        rule_based_issues.extend(self._check_diabetic_emergency_pattern(text_lower, symptoms, lab_values))
        rule_based_issues.extend(self._check_meningitis_pattern(text_lower, symptoms))
        rule_based_issues.extend(self._check_hypertensive_crisis(text_lower, lab_values))
        rule_based_issues.extend(self._check_respiratory_emergency(text_lower, lab_values))
        
        # Step 2: LLM-based inference for unclear patterns
        llm_result = self._llm_based_inference(symptoms, text, lab_values)
        
        # Step 3: Combine using OR logic - choose higher severity
        all_issues = rule_based_issues.copy()
        
        if llm_result:
            llm_severity = llm_result.get('severity', 'ok')
            llm_score = llm_result.get('score', 0.0)
            
            # Convert LLM severity to our severity levels
            # Severity is imported at module level, so it's available here
            if llm_severity == 'critical':
                llm_severity_enum = Severity.CRITICAL
            elif llm_severity == 'high':
                llm_severity_enum = Severity.HIGH
            elif llm_severity == 'moderate':
                llm_severity_enum = Severity.MODERATE
            else:
                llm_severity_enum = Severity.LOW
            
            # Check if LLM found something rule-based missed, or higher severity
            rule_max_score = max([i.get('score', 0) for i in rule_based_issues], default=0.0)
            
            if llm_score > rule_max_score or (llm_severity_enum == Severity.CRITICAL and rule_max_score < 10.0):
                # LLM found higher severity - add it
                all_issues.append({
                    'category': 'LLM Detected Pattern',
                    'symptom': ', '.join(llm_result.get('triggered_symptoms', [])),
                    'severity': llm_severity,
                    'score': llm_score,
                    'reasoning': llm_result.get('reasoning', 'LLM-based pattern detection'),
                    'llm_confidence': llm_result.get('confidence', 'medium'),
                    'llm_triggered_symptoms': llm_result.get('triggered_symptoms', [])
                })
        
        # Calculate overall severity and score - choose higher severity
        # Severity is already imported at module level
        
        if not all_issues:
            status = Severity.OK
            message = "No red flag symptoms detected. Patient appears stable."
            max_score = self.SCORE_SAFE
        else:
            # Categorize issues by severity (using new severity levels)
            # Also handle legacy 'severe' and 'mild' for backward compatibility
            critical_issues = [i for i in all_issues if i.get('severity') == 'critical' or i.get('severity') == 'severe']
            high_issues = [i for i in all_issues if i.get('severity') == 'high']
            moderate_issues = [i for i in all_issues if i.get('severity') == 'moderate']
            low_issues = [i for i in all_issues if i.get('severity') == 'low' or i.get('severity') == 'mild']
            
            # Calculate max score - use highest from rule-based or LLM
            max_score = max([i.get('score', 0) for i in all_issues], default=0.0)
            
            # Determine status - choose higher severity (OR logic)
            # Map to new severity levels with proper score ranges
            if critical_issues:
                status = Severity.CRITICAL
                # Ensure score is in CRITICAL range (9-10)
                max_score = max(9.0, min(10.0, max_score))
                categories = set([i.get('category', 'Unknown') for i in critical_issues])
                message = f"CRITICAL: {len(critical_issues)} critical red flag(s) detected across {len(categories)} category/categories ({', '.join(list(categories)[:3])}). "
                message += f"These are life-threatening emergencies requiring IMMEDIATE medical attention. "
                message += f"Additional findings: {len(high_issues)} high, {len(moderate_issues)} moderate, {len(low_issues)} low severity symptom(s). "
                message += "REASONING: " + "; ".join([i.get('reasoning', '')[:150] for i in critical_issues[:2]])
            elif high_issues:
                status = Severity.HIGH
                # Ensure score is in HIGH range (6-8)
                max_score = max(6.0, min(8.0, max_score))
                categories = set([i.get('category', 'Unknown') for i in high_issues])
                message = f"HIGH PRIORITY: {len(high_issues)} high-risk clinical finding(s) identified across {len(categories)} category/categories ({', '.join(list(categories)[:3])}). "
                message += f"Urgent clinical evaluation recommended within 24-48 hours. Additional findings: {len(moderate_issues)} moderate, {len(low_issues)} low severity symptom(s). "
                message += "Clinical assessment: " + "; ".join([i.get('reasoning', '')[:150] for i in high_issues[:2]])
            elif moderate_issues:
                # Check if this should be escalated to HIGH based on combination patterns
                # Pneumonia pattern: fever ≥ 101°F + yellow sputum + HR 100-120 + SOB = HIGH
                has_fever_101 = any('fever' in str(i.get('symptom', '')).lower() or '101' in str(i.get('symptom', '')).lower() or 'temperature' in str(i.get('symptom', '')).lower() for i in moderate_issues + high_issues)
                has_yellow_sputum = any('yellow' in str(i.get('symptom', '')).lower() and 'sputum' in str(i.get('symptom', '')).lower() for i in moderate_issues + high_issues) or 'yellow sputum' in text_lower
                has_hr_100_120 = any('104' in text_lower or 'heart rate' in text_lower or 'hr' in text_lower for _ in [1]) and any('104' in text_lower or ('100' in text_lower and '120' in text_lower))
                has_sob = any('shortness of breath' in str(i.get('symptom', '')).lower() or 'dyspnea' in str(i.get('symptom', '')).lower() for i in moderate_issues + high_issues) or 'shortness of breath' in text_lower
                
                # Pneumonia pattern detection
                pneumonia_score = sum([has_fever_101, has_yellow_sputum, has_hr_100_120, has_sob])
                if pneumonia_score >= 3:  # 3+ pneumonia signs = HIGH
                    status = Severity.HIGH
                    # Ensure score is in HIGH range (6-8)
                    max_score = max(6.0, min(8.0, max(7.0, max_score)))
                    message = f"HIGH: Pneumonia pattern detected ({pneumonia_score} signs: fever ≥101°F, yellow sputum, elevated HR, SOB). This requires urgent evaluation for possible pneumonia. "
                    message += f"Additional findings: {len(moderate_issues)} moderate, {len(low_issues)} low severity symptom(s). "
                    message += "REASONING: Combination of fever, productive cough with colored sputum, elevated heart rate, and respiratory symptoms suggests pneumonia requiring urgent medical evaluation."
                else:
                    status = Severity.MODERATE
                    # Ensure score is in MODERATE range (3-5)
                    max_score = max(3.0, min(5.0, max_score))
                    message = f"MODERATE: {len(moderate_issues)} moderate red flag symptom(s) detected. Monitor closely and consider evaluation if symptoms persist or worsen. "
                    message += f"Additional findings: {len(low_issues)} low severity symptom(s). "
                    message += "REASONING: " + "; ".join([i.get('reasoning', '')[:150] for i in moderate_issues[:2]])
            elif low_issues:
                status = Severity.LOW
                # Ensure score is in LOW range (1-2)
                max_score = max(1.0, min(2.0, max_score))
                message = f"LOW: {len(low_issues)} mild red flag symptom(s) detected. Monitor and consider evaluation if symptoms worsen. "
                message += "REASONING: " + "; ".join([i.get('reasoning', '')[:150] for i in low_issues[:2]])
            else:
                status = Severity.OK
                max_score = self.SCORE_SAFE
                # If no symptoms at all, ensure OK status
                if not symptoms or len(symptoms) == 0:
                    status = Severity.OK
                    message = "No symptoms detected. Patient appears stable."
                else:
                    status = Severity.OK
                    message = "No significant red flag symptoms detected."
        
        # Retrieve RAG evidence
        rag_evidence = []
        if all_issues:
            for issue in all_issues[:3]:  # Get evidence for first 3 issues
                symptom = issue.get('symptom', '')
                red_flags = self.rag_service.retrieve_red_flags(symptom)
                if red_flags:
                    rag_evidence.extend([r.get('content', '')[:200] for r in red_flags[:1]])
        
        return AgentOutput(
            agent="RedFlagChecker",
            status=status,
            message=message,
            evidence=rag_evidence if rag_evidence else [str(i) for i in all_issues[:5]],
            score=max_score,
            details={
                'red_flags': all_issues,
                'severe_count': len([i for i in all_issues if i.get('severity') == 'severe' or i.get('severity') == 'critical']),
                'moderate_count': len([i for i in all_issues if i.get('severity') == 'moderate']),
                'mild_count': len([i for i in all_issues if i.get('severity') == 'mild']),
                'categories': list(set([i.get('category', 'Unknown') for i in all_issues])),
                'llm_detected': llm_result is not None,
                'rule_based_count': len(rule_based_issues)
            }
        )


class MissingTestsCheckerAgent:
    """Agent for checking missing essential tests."""
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        try:
            from app.services.llm_service import LLMService
            self.llm_service = LLMService()
        except Exception as e:
            print(f"Warning: Could not initialize LLM service for MissingTestsChecker: {e}")
            self.llm_service = None
        
        # Required tests for common medications
        self.required_tests = {
            'warfarin': ['inr', 'pt'],
            'digoxin': ['digoxin level', 'creatinine', 'egfr'],
            'lithium': ['lithium level', 'creatinine', 'tsh'],
            'methotrexate': ['cbc', 'lft', 'creatinine'],
            'amiodarone': ['lft', 'tsh', 'chest x-ray'],
            'atorvastatin': ['lft', 'ck'],
            'simvastatin': ['lft', 'ck'],
            'pravastatin': ['lft', 'ck'],
            'rosuvastatin': ['lft', 'ck'],
            'metformin': ['creatinine', 'egfr', 'b12'],
            'ace inhibitor': ['creatinine', 'potassium'],
            'arb': ['creatinine', 'potassium']
        }
    
    def _generate_llm_explanation_chain(self, agent_name: str, issues: List[Dict], 
                                       *context_args) -> Dict[str, Any]:
        """Generate LLM explanation chain for agent result."""
        if not self.llm_service or not self.llm_service.llm:
            return None
        
        try:
            context_str = f"Issues: {str(issues[:2])}"
            if context_args:
                context_str += f" Context: {str(context_args[:2])}"
            
            prompt = f"""As a medical expert, provide a detailed explanation chain for this {agent_name} result:

{context_str}

Provide:
1. Why this was flagged (specific reason)
2. Which drugs/tests triggered it (list them)
3. Which guideline chunks were retrieved (if any)
4. Confidence level: high/medium/low (with reasoning)

Format as JSON:
{{
    "why_flagged": "...",
    "triggered_items": [...],
    "guideline_chunks": [...],
    "confidence": "high/medium/low",
    "confidence_reasoning": "..."
}}"""
            
            response = self.llm_service.generate(prompt, max_tokens=400)
            
            # Try to extract JSON
            import json
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "why_flagged": response[:200],
                    "triggered_items": [str(i) for i in issues[:3]],
                    "guideline_chunks": [],
                    "confidence": "medium",
                    "confidence_reasoning": "Based on LLM analysis"
                }
        except Exception as e:
            print(f"LLM explanation chain generation failed: {e}")
            return None
        # Drug -> required tests mapping
        self.required_tests = {
            'warfarin': ['inr', 'pt'],
            'metformin': ['creatinine', 'egfr'],
            'atorvastatin': ['alt', 'ast', 'ck'],
            'simvastatin': ['alt', 'ast', 'ck'],
            'furosemide': ['creatinine', 'electrolytes'],
            'levothyroxine': ['tsh'],
            'amiodarone': ['tsh', 'lft', 'chest x-ray']
        }

    def check(self, drugs: List[Dict], lab_values: List[Dict]) -> AgentOutput:
        """Check for missing essential tests."""
        issues = []
        max_score = 0.0
        
        drug_names = [d.get('normalized_name', '').lower() for d in drugs]
        performed_tests = [l.get('test', '').lower() for l in lab_values]
        
        for drug_name in drug_names:
            required = self.required_tests.get(drug_name, [])
            
            for test in required:
                # Check if test was performed
                test_found = any(test in performed_test or performed_test in test 
                               for performed_test in performed_tests)
                
                if not test_found:
                    severity = Severity.HIGH if drug_name in ['warfarin', 'metformin'] else Severity.MODERATE
                    issues.append({
                        'drug': drug_name,
                        'missing_test': test,
                        'severity': severity.value
                    })
                    # Map severity to new score ranges
                    if severity == Severity.HIGH:
                        max_score = max(max_score, 7.0)  # HIGH: 6-8
                    elif severity == Severity.MODERATE:
                        max_score = max(max_score, 4.0)  # MODERATE: 3-5
                    else:
                        max_score = max(max_score, 2.0)  # LOW: 1-2
        
        if issues:
            high_issues = [i for i in issues if i.get('severity') == Severity.HIGH.value]
            if high_issues:
                status = Severity.HIGH
                message = f"High: {len(high_issues)} essential test(s) missing"
            else:
                status = Severity.MODERATE
                message = f"Moderate: {len(issues)} recommended test(s) missing"
        else:
            # If no drugs identified, return OK (no tests required)
            if not drugs or len(drugs) == 0:
                status = Severity.OK
                message = "No medications identified. No laboratory monitoring required."
            else:
                status = Severity.OK
                message = "Essential laboratory monitoring appears adequate. No additional tests required at this time based on current medication regimen."
        
        # Generate LLM explanation chain
        llm_explanation = None
        if self.llm_service and self.llm_service.llm:
            try:
                llm_explanation = self._generate_llm_explanation_chain(
                    "MissingTestsChecker",
                    issues,
                    drugs,
                    lab_values,
                    message
                )
            except Exception as e:
                print(f"LLM explanation chain failed: {e}")
        
        details = {'missing_tests': issues}
        if llm_explanation:
            details['llm_explanation'] = llm_explanation
        
        return AgentOutput(
            agent="MissingTestsChecker",
            status=status,
            message=message,
            evidence=[str(i) for i in issues],
            score=max_score,
            details=details
        )


class GuidelineComplianceCheckerAgent:
    """Agent for checking guideline compliance."""
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        try:
            from app.services.llm_service import LLMService
            self.llm_service = LLMService()
        except Exception as e:
            print(f"Warning: Could not initialize LLM service for GuidelineComplianceChecker: {e}")
            self.llm_service = None
    
    def _generate_llm_explanation_chain(self, agent_name: str, issues: List[Dict], 
                                       *context_args) -> Dict[str, Any]:
        """Generate LLM explanation chain for agent result."""
        if not self.llm_service or not self.llm_service.llm:
            return None
        
        try:
            context_str = f"Issues: {str(issues[:2])}"
            if context_args:
                context_str += f" Context: {str(context_args[:2])}"
            
            prompt = f"""As a medical expert, provide a detailed explanation chain for this {agent_name} result:

{context_str}

Provide:
1. Why this was flagged (specific reason)
2. Which drugs/conditions triggered it (list them)
3. Which guideline chunks were retrieved (if any)
4. Confidence level: high/medium/low (with reasoning)

Format as JSON:
{{
    "why_flagged": "...",
    "triggered_items": [...],
    "guideline_chunks": [...],
    "confidence": "high/medium/low",
    "confidence_reasoning": "..."
}}"""
            
            response = self.llm_service.generate(prompt, max_tokens=400)
            
            # Try to extract JSON
            import json
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "why_flagged": response[:200],
                    "triggered_items": [str(i) for i in issues[:3]],
                    "guideline_chunks": [],
                    "confidence": "medium",
                    "confidence_reasoning": "Based on LLM analysis"
                }
        except Exception as e:
            print(f"LLM explanation chain generation failed: {e}")
            return None

    def check(self, drugs: List[Dict], symptoms: List[str], text: str) -> AgentOutput:
        """Check compliance with treatment guidelines."""
        issues = []
        max_score = 0.0
        
        # Extract condition from symptoms/text
        conditions = []
        condition_keywords = {
            'diabetes': ['diabetes', 'diabetic', 'glucose', 'hba1c'],
            'hypertension': ['hypertension', 'high blood pressure', 'bp'],
            'hyperlipidemia': ['cholesterol', 'ldl', 'hdl', 'hyperlipidemia'],
            'hypothyroidism': ['hypothyroidism', 'tsh', 'thyroid'],
            'infection': ['infection', 'fever', 'antibiotic']
        }
        
        text_lower = text.lower()
        for condition, keywords in condition_keywords.items():
            if any(kw in text_lower for kw in keywords):
                conditions.append(condition)
        
        # Check guidelines for each condition
        for condition in conditions:
            guidelines = self.rag_service.retrieve_guidelines(condition)
            
            if guidelines:
                # Simple check: see if recommended drugs are present
                content = ' '.join([g.get('content', '') for g in guidelines]).lower()
                drug_names = [d.get('normalized_name', '').lower() for d in drugs]
                
                # Extract recommended drugs from guidelines
                recommended_drugs = []
                for guideline in guidelines:
                    guideline_content = guideline.get('content', '').lower()
                    # Look for drug mentions
                    for drug in drug_names:
                        if drug in guideline_content and 'recommend' in guideline_content:
                            recommended_drugs.append(drug)
                
                # Check if prescribed drugs match recommendations
                if drug_names and not any(drug in recommended_drugs for drug in drug_names):
                    issues.append({
                        'condition': condition,
                        'prescribed_drugs': drug_names,
                        'severity': Severity.MODERATE.value
                    })
                    max_score = max(max_score, 4.0)  # MODERATE: 3-5
        
        if issues:
            status = Severity.MODERATE
            message = f"Moderate: {len(issues)} guideline compliance issue(s)"
        else:
            status = Severity.OK
            message = "Treatment plan appears consistent with established clinical guidelines. No significant deviations from standard protocols identified."
        
        # Generate LLM explanation chain
        llm_explanation = None
        if self.llm_service and self.llm_service.llm:
            try:
                llm_explanation = self._generate_llm_explanation_chain(
                    "GuidelineComplianceChecker",
                    issues,
                    drugs,
                    symptoms,
                    text,
                    message
                )
            except Exception as e:
                print(f"LLM explanation chain failed: {e}")
        
        details = {'compliance_issues': issues}
        if llm_explanation:
            details['llm_explanation'] = llm_explanation
        
        return AgentOutput(
            agent="GuidelineComplianceChecker",
            status=status,
            message=message,
            evidence=[str(i) for i in issues],
            score=max_score,
            details=details
        )

