"""
Advanced Medical Pattern Detector for emergency conditions.
Detects critical medical emergencies based on symptom combinations.
"""
from typing import List, Dict, Any, Set
import re


class AdvancedPatternDetector:
    """Detects critical medical emergencies based on established patterns."""
    
    def __init__(self):
        self._init_emergency_patterns()
    
    def _init_emergency_patterns(self):
        """Initialize emergency detection patterns."""
        
        # Myocardial Infarction (MI) patterns
        self.mi_patterns = {
            'required': ['chest pain'],
            'supporting': ['sweating', 'diaphoresis', 'shortness of breath', 'dyspnea', 
                          'nausea', 'vomiting', 'radiating pain', 'arm pain', 'jaw pain',
                          'dizziness', 'lightheadedness', 'syncope'],
            'min_supporting': 2  # Need at least 2 supporting symptoms
        }
        
        # Stroke patterns
        self.stroke_patterns = {
            'required': [],  # Any of the following
            'critical': ['facial droop', 'arm weakness', 'speech difficulty', 
                        'slurred speech', 'aphasia', 'hemiparesis'],
            'supporting': ['headache', 'dizziness', 'confusion', 'vision changes',
                          'loss of balance', 'numbness', 'tingling'],
            'min_critical': 1,  # Need at least 1 critical symptom
            'min_total': 2  # Or 2 supporting symptoms
        }
        
        # Meningitis patterns
        self.meningitis_patterns = {
            'required': ['fever'],
            'supporting': ['headache', 'neck stiffness', 'nuchal rigidity', 
                          'photophobia', 'light sensitivity', 'vomiting', 
                          'nausea', 'confusion', 'altered mental status'],
            'min_supporting': 3  # Need at least 3 supporting symptoms
        }
        
        # Sepsis patterns
        self.sepsis_patterns = {
            'required': ['fever'],  # Or hypothermia
            'supporting': ['chills', 'rigors', 'shaking chills', 'sweating',
                          'confusion', 'altered mental status', 'rapid breathing',
                          'tachypnea', 'rapid heart rate', 'tachycardia',
                          'low blood pressure', 'hypotension'],
            'min_supporting': 3  # Need at least 3 supporting symptoms
        }
        
        # DKA / HHS patterns
        self.dka_hhs_patterns = {
            'required': [],  # Glucose check or symptoms
            'critical_labs': ['glucose > 250', 'glucose > 300', 'glucose > 350'],
            'symptoms': ['polyuria', 'excessive urination', 'polydipsia', 
                        'excessive thirst', 'blurred vision', 'nausea', 'vomiting',
                        'dehydration', 'dry mouth', 'ketones', 'ketonuria',
                        'abdominal pain', 'kussmaul breathing'],
            'min_symptoms': 3  # Need at least 3 symptoms
        }
        
        # Hypertensive crisis patterns
        self.hypertensive_crisis_patterns = {
            'required': [],  # BP check
            'critical_bp': ['bp > 180/120', 'blood pressure > 180/120'],
            'symptoms': ['headache', 'severe headache', 'chest pain', 
                        'shortness of breath', 'dyspnea', 'vision changes',
                        'blurred vision', 'confusion', 'nausea', 'vomiting'],
            'min_symptoms': 1  # Need BP + at least 1 symptom
        }
        
        # Respiratory failure patterns
        self.respiratory_failure_patterns = {
            'required': [],  # O2 sat or symptoms
            'critical_labs': ['oxygen < 92', 'spo2 < 92', 'o2 sat < 92'],
            'symptoms': ['severe shortness of breath', 'unable to breathe',
                        'respiratory distress', 'gasping', 'cyanosis', 
                        'blue lips', 'blue skin', 'wheezing', 'stridor'],
            'min_symptoms': 2  # Need O2 sat OR 2+ symptoms
        }
    
    def detect_emergencies(self, symptoms: List[str], text: str, 
                          lab_values: List[Any] = None, 
                          vitals: List[Any] = None) -> List[Dict[str, Any]]:
        """Detect all matching emergency patterns."""
        detected = []
        text_lower = text.lower()
        symptoms_lower = [s.lower() for s in symptoms]
        symptoms_set = set(symptoms_lower)
        
        # Check each emergency pattern
        mi_result = self._check_mi(symptoms_set, text_lower)
        if mi_result:
            detected.append(mi_result)
        
        stroke_result = self._check_stroke(symptoms_set, text_lower)
        if stroke_result:
            detected.append(stroke_result)
        
        meningitis_result = self._check_meningitis(symptoms_set, text_lower)
        if meningitis_result:
            detected.append(meningitis_result)
        
        sepsis_result = self._check_sepsis(symptoms_set, text_lower, vitals)
        if sepsis_result:
            detected.append(sepsis_result)
        
        dka_result = self._check_dka_hhs(symptoms_set, text_lower, lab_values)
        if dka_result:
            detected.append(dka_result)
        
        hypertensive_result = self._check_hypertensive_crisis(symptoms_set, text_lower, vitals, lab_values)
        if hypertensive_result:
            detected.append(hypertensive_result)
        
        respiratory_result = self._check_respiratory_failure(symptoms_set, text_lower, vitals, lab_values)
        if respiratory_result:
            detected.append(respiratory_result)
        
        return detected
    
    def _check_mi(self, symptoms_set: Set[str], text_lower: str) -> Dict[str, Any]:
        """Check for Myocardial Infarction pattern."""
        patterns = self.mi_patterns
        
        # Check required symptom
        has_required = any(pattern in text_lower or pattern in symptoms_set 
                          for pattern in patterns['required'])
        
        if not has_required:
            return None
        
        # Count supporting symptoms
        supporting_count = sum(1 for pattern in patterns['supporting']
                              if pattern in text_lower or pattern in symptoms_set)
        
        if supporting_count >= patterns['min_supporting']:
            return {
                'emergency': 'Myocardial Infarction',
                'severity': 'critical',
                'score': 10.0,
                'matched_symptoms': supporting_count + 1,
                'reasoning': f'MI pattern detected: chest pain with {supporting_count} supporting symptoms (sweating, SOB, nausea, etc.). This is a medical emergency requiring immediate ECG, cardiac enzymes, and possible reperfusion therapy.',
                'override_agent_score': True
            }
        
        return None
    
    def _check_stroke(self, symptoms_set: Set[str], text_lower: str) -> Dict[str, Any]:
        """Check for Stroke pattern."""
        patterns = self.stroke_patterns
        
        # Check critical symptoms
        critical_count = sum(1 for pattern in patterns['critical']
                            if pattern in text_lower or pattern in symptoms_set)
        
        # Check supporting symptoms
        supporting_count = sum(1 for pattern in patterns['supporting']
                              if pattern in text_lower or pattern in symptoms_set)
        
        if critical_count >= patterns['min_critical']:
            return {
                'emergency': 'Stroke',
                'severity': 'critical',
                'score': 10.0,
                'matched_symptoms': critical_count + supporting_count,
                'reasoning': f'STROKE pattern detected: {critical_count} critical symptom(s) (facial droop, arm weakness, speech difficulty). This is a neurological emergency requiring immediate CT/MRI, stroke protocol activation, and possible thrombolytics.',
                'override_agent_score': True
            }
        elif supporting_count >= patterns['min_total']:
            return {
                'emergency': 'Possible Stroke',
                'severity': 'critical',
                'score': 9.0,
                'matched_symptoms': supporting_count,
                'reasoning': f'Possible stroke pattern: {supporting_count} neurological symptoms. Requires urgent neurological evaluation and imaging.',
                'override_agent_score': True
            }
        
        return None
    
    def _check_meningitis(self, symptoms_set: Set[str], text_lower: str) -> Dict[str, Any]:
        """Check for Meningitis pattern."""
        patterns = self.meningitis_patterns
        
        # Check required symptom
        has_required = any(pattern in text_lower or pattern in symptoms_set 
                          for pattern in patterns['required'])
        
        if not has_required:
            return None
        
        # Count supporting symptoms
        supporting_count = sum(1 for pattern in patterns['supporting']
                              if pattern in text_lower or pattern in symptoms_set)
        
        if supporting_count >= patterns['min_supporting']:
            return {
                'emergency': 'Meningitis',
                'severity': 'critical',
                'score': 10.0,
                'matched_symptoms': supporting_count + 1,
                'reasoning': f'Meningitis pattern detected: fever with {supporting_count} supporting symptoms (headache, neck stiffness, photophobia, vomiting). This is a medical emergency requiring immediate lumbar puncture, blood cultures, and antibiotic therapy.',
                'override_agent_score': True
            }
        
        return None
    
    def _check_sepsis(self, symptoms_set: Set[str], text_lower: str, 
                     vitals: List[Dict] = None) -> Dict[str, Any]:
        """Check for Sepsis pattern."""
        patterns = self.sepsis_patterns
        
        # Check for fever or hypothermia
        has_fever = any(pattern in text_lower or pattern in symptoms_set 
                       for pattern in ['fever', 'pyrexia', 'hyperthermia'])
        has_hypothermia = any(pattern in text_lower for pattern in ['hypothermia', 'low temperature', 'temp < 95'])
        
        if not (has_fever or has_hypothermia):
            return None
        
        # Count supporting symptoms
        supporting_count = sum(1 for pattern in patterns['supporting']
                              if pattern in text_lower or pattern in symptoms_set)
        
        # Check vitals for rapid HR, RR, or low BP
        if vitals:
            for vital in vitals:
                # Handle both dict and string formats
                if isinstance(vital, dict):
                    vital_text = str(vital.get('text', '')).lower()
                    hr_value = vital.get('value')
                    rr_value = vital.get('value')
                else:
                    vital_text = str(vital).lower()
                    hr_value = None
                    rr_value = None
                
                if 'heart rate' in vital_text or 'hr' in vital_text or 'pulse' in vital_text:
                    if not hr_value:
                        hr_value = self._extract_number(vital_text)
                    if hr_value and hr_value > 90:
                        supporting_count += 1
                if 'respiratory rate' in vital_text or 'rr' in vital_text:
                    if not rr_value:
                        rr_value = self._extract_number(vital_text)
                    if rr_value and rr_value > 20:
                        supporting_count += 1
        
        if supporting_count >= patterns['min_supporting']:
            return {
                'emergency': 'Sepsis',
                'severity': 'critical',
                'score': 10.0,
                'matched_symptoms': supporting_count + 1,
                'reasoning': f'SEPSIS pattern detected: fever/hypothermia with {supporting_count} supporting symptoms. This is a life-threatening emergency requiring immediate antibiotics, fluid resuscitation, and ICU monitoring.',
                'override_agent_score': True
            }
        
        return None
    
    def _check_dka_hhs(self, symptoms_set: Set[str], text_lower: str,
                      lab_values: List[Dict] = None) -> Dict[str, Any]:
        """Check for DKA/HHS pattern."""
        patterns = self.dka_hhs_patterns
        
        # Check for critical glucose levels
        has_critical_glucose = False
        glucose_value = None
        
        if lab_values:
            for lab in lab_values:
                # Handle both dict and string formats
                if isinstance(lab, dict):
                    lab_text = str(lab.get('text', '')).lower()
                    glucose_value = lab.get('value')
                else:
                    lab_text = str(lab).lower()
                    glucose_value = None
                
                if 'glucose' in lab_text or 'blood sugar' in lab_text:
                    if not glucose_value:
                        glucose_value = self._extract_number(lab_text)
                    if glucose_value and glucose_value > 250:
                        has_critical_glucose = True
                        break
        
        # Also check text
        if not has_critical_glucose:
            for pattern in patterns['critical_labs']:
                if pattern in text_lower:
                    match = re.search(r'(\d+(?:\.\d+)?)', text_lower)
                    if match:
                        glucose_value = float(match.group(1))
                        if glucose_value > 250:
                            has_critical_glucose = True
                            break
        
        # Count symptoms
        symptom_count = sum(1 for pattern in patterns['symptoms']
                           if pattern in text_lower or pattern in symptoms_set)
        
        if has_critical_glucose and symptom_count >= patterns['min_symptoms']:
            return {
                'emergency': 'DKA/HHS',
                'severity': 'critical',
                'score': 10.0,
                'matched_symptoms': symptom_count + 1,
                'reasoning': f'DKA/HHS pattern detected: glucose >250 mg/dL ({glucose_value} mg/dL) with {symptom_count} diabetic symptoms. This is a medical emergency requiring immediate insulin therapy, fluid resuscitation, and monitoring for acidosis.',
                'override_agent_score': True
            }
        elif symptom_count >= patterns['min_symptoms'] + 1:  # More symptoms without confirmed glucose
            return {
                'emergency': 'Possible DKA/HHS',
                'severity': 'critical',
                'score': 9.0,
                'matched_symptoms': symptom_count,
                'reasoning': f'Possible DKA/HHS: {symptom_count} diabetic symptoms. Check glucose and ketones immediately.',
                'override_agent_score': True
            }
        
        return None
    
    def _check_hypertensive_crisis(self, symptoms_set: Set[str], text_lower: str,
                                   vitals: List[Dict] = None,
                                   lab_values: List[Dict] = None) -> Dict[str, Any]:
        """Check for Hypertensive Crisis pattern."""
        patterns = self.hypertensive_crisis_patterns
        
        # Check BP
        bp_systolic = None
        bp_diastolic = None
        
        if vitals:
            for vital in vitals:
                # Handle both dict and string formats
                if isinstance(vital, dict):
                    vital_text = str(vital.get('text', '')).lower()
                else:
                    vital_text = str(vital).lower()
                
                if 'bp' in vital_text or 'blood pressure' in vital_text:
                    match = re.search(r'(\d+)\s*[/-]\s*(\d+)', vital_text)
                    if match:
                        bp_systolic = int(match.group(1))
                        bp_diastolic = int(match.group(2))
        
        # Also check text
        if not bp_systolic:
            match = re.search(r'(?:bp|blood pressure)[:\s]*(\d+)\s*[/-]\s*(\d+)', text_lower)
            if match:
                bp_systolic = int(match.group(1))
                bp_diastolic = int(match.group(2))
        
        if bp_systolic and bp_diastolic and (bp_systolic > 180 or bp_diastolic > 120):
            symptom_count = sum(1 for pattern in patterns['symptoms']
                              if pattern in text_lower or pattern in symptoms_set)
            
            if symptom_count >= patterns['min_symptoms']:
                return {
                    'emergency': 'Hypertensive Crisis',
                    'severity': 'critical',
                    'score': 10.0,
                    'matched_symptoms': symptom_count + 1,
                    'reasoning': f'Hypertensive crisis detected: BP {bp_systolic}/{bp_diastolic} mmHg with {symptom_count} symptom(s). This is a medical emergency requiring immediate BP reduction with IV antihypertensives in monitored setting.',
                    'override_agent_score': True
                }
        
        return None
    
    def _check_respiratory_failure(self, symptoms_set: Set[str], text_lower: str,
                                  vitals: List[Dict] = None,
                                  lab_values: List[Dict] = None) -> Dict[str, Any]:
        """Check for Respiratory Failure pattern."""
        patterns = self.respiratory_failure_patterns
        
        # Check O2 saturation
        has_low_o2 = False
        o2_value = None
        
        if vitals:
            for vital in vitals:
                # Handle both dict and string formats
                if isinstance(vital, dict):
                    vital_text = str(vital.get('text', '')).lower()
                    o2_value = vital.get('value')
                else:
                    vital_text = str(vital).lower()
                    o2_value = None
                
                if 'oxygen' in vital_text or 'o2' in vital_text or 'spo2' in vital_text:
                    if not o2_value:
                        o2_value = self._extract_number(vital_text)
                    if o2_value and o2_value < 92:
                        has_low_o2 = True
                        break
        
        # Also check text
        if not has_low_o2:
            match = re.search(r'(?:oxygen|o2|spo2|saturation)[:\s]*(\d+(?:\.\d+)?)\s*(?:%|percent)', text_lower)
            if match:
                o2_value = float(match.group(1))
                if o2_value < 92:
                    has_low_o2 = True
        
        # Count symptoms
        symptom_count = sum(1 for pattern in patterns['symptoms']
                           if pattern in text_lower or pattern in symptoms_set)
        
        if has_low_o2:
            return {
                'emergency': 'Respiratory Failure',
                'severity': 'critical',
                'score': 10.0,
                'matched_symptoms': symptom_count + 1,
                'reasoning': f'Respiratory failure detected: O2 saturation {o2_value}% (< 92%) with {symptom_count} respiratory symptom(s). This is a life-threatening emergency requiring immediate oxygen therapy and respiratory support.',
                'override_agent_score': True
            }
        elif symptom_count >= patterns['min_symptoms']:
            return {
                'emergency': 'Respiratory Distress',
                'severity': 'critical',
                'score': 9.0,
                'matched_symptoms': symptom_count,
                'reasoning': f'Respiratory distress pattern: {symptom_count} severe respiratory symptoms. Check O2 saturation immediately and prepare for respiratory support.',
                'override_agent_score': True
            }
        
        return None
    
    def _extract_number(self, text: str) -> float:
        """Extract first number from text."""
        match = re.search(r'(\d+(?:\.\d+)?)', text)
        return float(match.group(1)) if match else None

