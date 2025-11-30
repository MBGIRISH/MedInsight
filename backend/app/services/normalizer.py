import re
from typing import Dict, Any, List
from app.models.schemas import ExtractedEntity, EntityType, NERResult


class NormalizerService:
    """Service for normalizing medical entities to standard formats."""
    
    # Frequency normalization map
    FREQUENCY_MAP = {
        'qd': 'once daily',
        'bid': 'twice daily',
        'tid': 'three times daily',
        'qid': 'four times daily',
        'once': 'once daily',
        'twice': 'twice daily',
        'daily': 'once daily',
        'weekly': 'once weekly',
        'monthly': 'once monthly'
    }
    
    # Unit conversion factors (to mg)
    UNIT_CONVERSION = {
        'g': 1000,  # grams to mg
        'mg': 1,
        'mcg': 0.001,
        'ml': 1,  # Assuming 1ml = 1mg for liquid medications
        'units': 1  # Placeholder
    }

    @staticmethod
    def normalize_entities(entities: List[ExtractedEntity], text: str) -> Dict[str, Any]:
        """Normalize all extracted entities."""
        normalized = {
            'drugs': [],
            'dosages': [],
            'frequencies': [],
            'durations': [],
            'symptoms': [],
            'lab_values': [],
            'vitals': []
        }
        
        # Group entities by type
        drugs = [e for e in entities if e.type == EntityType.DRUG]
        dosages = [e for e in entities if e.type == EntityType.DOSAGE]
        frequencies = [e for e in entities if e.type == EntityType.FREQUENCY]
        durations = [e for e in entities if e.type == EntityType.DURATION]
        symptoms = [e for e in entities if e.type == EntityType.SYMPTOM]
        lab_values = [e for e in entities if e.type == EntityType.LAB_VALUE]
        vitals = [e for e in entities if e.type == EntityType.VITALS]
        
        # Normalize drugs
        normalized['drugs'] = [NormalizerService._normalize_drug(e.text) for e in drugs]
        
        # Normalize dosages
        normalized['dosages'] = [NormalizerService._normalize_dosage(e.text) for e in dosages]
        
        # Normalize frequencies
        normalized['frequencies'] = [NormalizerService._normalize_frequency(e.text) for e in frequencies]
        
        # Normalize durations
        normalized['durations'] = [NormalizerService._normalize_duration(e.text) for e in durations]
        
        # Normalize symptoms
        normalized['symptoms'] = [e.text.lower().strip() for e in symptoms]
        
        # Normalize lab values
        normalized['lab_values'] = [NormalizerService._normalize_lab_value(e.text) for e in lab_values]
        
        # Normalize vitals
        normalized['vitals'] = [NormalizerService._normalize_vitals(e.text) for e in vitals]
        
        return normalized

    @staticmethod
    def _normalize_drug(drug_text: str) -> Dict[str, Any]:
        """Normalize drug name."""
        return {
            'name': drug_text.strip().title(),
            'normalized_name': drug_text.lower().strip()
        }

    @staticmethod
    def _normalize_dosage(dosage_text: str) -> Dict[str, Any]:
        """Normalize dosage to mg."""
        dosage_text = dosage_text.lower().strip()
        
        # Handle fractions
        if '/' in dosage_text:
            parts = dosage_text.split()
            for part in parts:
                if '/' in part:
                    num, den = part.split('/')
                    value = float(num) / float(den)
                    # Find unit
                    unit = 'tablet' if 'tablet' in dosage_text else 'mg'
                    return {
                        'original': dosage_text,
                        'value_mg': value * NormalizerService.UNIT_CONVERSION.get(unit, 1),
                        'unit': 'mg',
                        'form': 'tablet' if 'tablet' in dosage_text else 'mg'
                    }
        
        # Extract number and unit
        match = re.search(r'(\d+(?:\.\d+)?)\s*(mg|g|mcg|ml|tablets?|capsules?|units?)', dosage_text)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower().rstrip('s')
            
            # Convert to mg
            if unit in NormalizerService.UNIT_CONVERSION:
                value_mg = value * NormalizerService.UNIT_CONVERSION[unit]
            else:
                value_mg = value  # Default
            
            return {
                'original': dosage_text,
                'value_mg': value_mg,
                'unit': 'mg',
                'form': unit
            }
        
        # Handle word-based dosages
        word_map = {
            'one': 1,
            'two': 2,
            'three': 3,
            'four': 4,
            'five': 5,
            'half': 0.5,
            'quarter': 0.25
        }
        
        for word, num in word_map.items():
            if word in dosage_text:
                unit = 'tablet' if 'tablet' in dosage_text else 'mg'
                return {
                    'original': dosage_text,
                    'value_mg': num * NormalizerService.UNIT_CONVERSION.get(unit, 1),
                    'unit': 'mg',
                    'form': unit
                }
        
        return {
            'original': dosage_text,
            'value_mg': 0,
            'unit': 'unknown',
            'form': 'unknown'
        }

    @staticmethod
    def _normalize_frequency(freq_text: str) -> Dict[str, Any]:
        """Normalize frequency to standard format."""
        freq_lower = freq_text.lower().strip()
        
        # Check direct mapping
        if freq_lower in NormalizerService.FREQUENCY_MAP:
            normalized = NormalizerService.FREQUENCY_MAP[freq_lower]
        else:
            # Extract number and period
            match = re.search(r'(\d+)\s*(times?|x)\s*(per|a|daily|week|day)', freq_lower)
            if match:
                times = int(match.group(1))
                period = match.group(3)
                if period in ['daily', 'day', 'a']:
                    if times == 1:
                        normalized = 'once daily'
                    elif times == 2:
                        normalized = 'twice daily'
                    else:
                        normalized = f'{times} times daily'
                else:
                    normalized = f'{times} times weekly'
            else:
                normalized = freq_lower
        
        return {
            'original': freq_text,
            'normalized': normalized,
            'times_per_day': NormalizerService._extract_times_per_day(normalized)
        }

    @staticmethod
    def _extract_times_per_day(normalized_freq: str) -> int:
        """Extract number of times per day from normalized frequency."""
        if 'once' in normalized_freq:
            return 1
        elif 'twice' in normalized_freq:
            return 2
        elif 'three times' in normalized_freq:
            return 3
        elif 'four times' in normalized_freq:
            return 4
        else:
            match = re.search(r'(\d+)', normalized_freq)
            return int(match.group(1)) if match else 1

    @staticmethod
    def _normalize_duration(duration_text: str) -> Dict[str, Any]:
        """Normalize duration to days."""
        match = re.search(r'(\d+)\s*(days?|weeks?|months?|years?)', duration_text.lower())
        if match:
            value = int(match.group(1))
            unit = match.group(2).lower().rstrip('s')
            
            # Convert to days
            if unit == 'day':
                days = value
            elif unit == 'week':
                days = value * 7
            elif unit == 'month':
                days = value * 30
            elif unit == 'year':
                days = value * 365
            else:
                days = value
            
            return {
                'original': duration_text,
                'days': days,
                'weeks': round(days / 7, 1),
                'months': round(days / 30, 1)
            }
        
        return {
            'original': duration_text,
            'days': 0,
            'weeks': 0,
            'months': 0
        }

    @staticmethod
    def _normalize_lab_value(lab_text: str) -> Dict[str, Any]:
        """Normalize lab values."""
        lab_lower = lab_text.lower()
        
        # Extract test name and value
        test_match = re.search(r'(glucose|creatinine|hemoglobin|hba1c|cholesterol|ldl|hdl|triglycerides|alt|ast|tsh|t3|t4)', lab_lower)
        value_match = re.search(r'(\d+(?:\.\d+)?)\s*(mg/dl|mmol/l|units?)', lab_lower)
        
        if test_match and value_match:
            return {
                'test': test_match.group(1),
                'value': float(value_match.group(1)),
                'unit': value_match.group(2),
                'original': lab_text
            }
        
        return {
            'test': 'unknown',
            'value': 0,
            'unit': 'unknown',
            'original': lab_text
        }

    @staticmethod
    def _normalize_vitals(vitals_text: str) -> Dict[str, Any]:
        """Normalize vital signs."""
        vitals_lower = vitals_text.lower()
        
        # Blood pressure
        bp_match = re.search(r'(\d+)\s*/\s*(\d+)', vitals_text)
        if bp_match:
            return {
                'type': 'blood_pressure',
                'systolic': int(bp_match.group(1)),
                'diastolic': int(bp_match.group(2)),
                'original': vitals_text
            }
        
        return {
            'type': 'unknown',
            'original': vitals_text
        }

