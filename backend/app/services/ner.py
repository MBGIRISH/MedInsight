from typing import List, Dict, Any, Set
import re
from transformers import pipeline
from app.models.schemas import ExtractedEntity, EntityType, NERResult


class NERService:
    """Enhanced Medical Named Entity Recognition service with comprehensive symptom dictionaries and regex patterns."""
    
    def __init__(self):
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple"
            )
        except Exception as e:
            print(f"Warning: Could not load NER model: {e}")
            self.ner_pipeline = None
        
        # Initialize symptom dictionaries
        self._init_symptom_dictionaries()
    
    def _init_symptom_dictionaries(self):
        """Initialize comprehensive symptom dictionaries by category."""
        
        # Cardiac symptoms
        self.cardiac_symptoms = {
            'chest pain', 'chest discomfort', 'chest pressure', 'chest tightness',
            'crushing chest pain', 'angina', 'anginal pain', 'retrosternal pain',
            'heart attack', 'myocardial infarction', 'mi', 'acute coronary syndrome',
            'radiating pain', 'pain radiating to arm', 'pain radiating to jaw',
            'pain radiating to back', 'pain radiating to shoulder',
            'sweating', 'diaphoresis', 'profuse sweating', 'cold sweat',
            'palpitations', 'irregular heartbeat', 'heart racing', 'rapid heartbeat',
            'arrhythmia', 'atrial fibrillation', 'afib', 'tachycardia', 'bradycardia',
            'dizziness', 'lightheadedness', 'syncope', 'fainting', 'loss of consciousness',
            'shortness of breath', 'dyspnea', 'breathlessness', 'sob',
            'fatigue', 'weakness', 'lethargy', 'exertional fatigue',
            'nausea', 'vomiting', 'indigestion', 'heartburn'
        }
        
        # Neurological symptoms
        self.neurological_symptoms = {
            'headache', 'cephalgia', 'migraine', 'head pain', 'severe headache',
            'thunderclap headache', 'worst headache of life', 'worst headache',
            'stroke', 'cva', 'cerebrovascular accident', 'transient ischemic attack', 'tia',
            'loss of consciousness', 'unconscious', 'coma', 'altered consciousness',
            'seizure', 'convulsion', 'fit', 'epileptic seizure', 'status epilepticus',
            'facial droop', 'facial weakness', 'facial asymmetry',
            'arm weakness', 'leg weakness', 'limb weakness', 'hemiparesis',
            'speech difficulty', 'slurred speech', 'aphasia', 'dysarthria',
            'numbness', 'tingling', 'paresthesia', 'numbness in arm', 'numbness in leg',
            'confusion', 'disorientation', 'altered mental status', 'mental confusion',
            'memory loss', 'amnesia', 'cognitive decline',
            'dizziness', 'vertigo', 'unsteadiness', 'loss of balance',
            'vision changes', 'blurred vision', 'double vision', 'diplopia', 'visual field defect',
            'neck stiffness', 'nuchal rigidity', 'stiff neck',
            'photophobia', 'light sensitivity', 'sensitivity to light'
        }
        
        # Diabetic symptoms
        self.diabetic_symptoms = {
            'polyuria', 'excessive urination', 'frequent urination', 'increased urination',
            'polydipsia', 'excessive thirst', 'increased thirst', 'unquenchable thirst',
            'polyphagia', 'excessive hunger', 'increased appetite',
            'blurred vision', 'blurry vision', 'vision changes', 'visual disturbance',
            'nausea', 'vomiting', 'nausea and vomiting',
            'dehydration', 'dehydrated', 'dry mouth', 'dry skin', 'dry mucous membranes',
            'fatigue', 'weakness', 'lethargy', 'tiredness',
            'weight loss', 'unintended weight loss', 'rapid weight loss',
            'slow healing', 'poor wound healing', 'delayed healing',
            'frequent infections', 'recurrent infections',
            'ketoacidosis', 'dka', 'diabetic ketoacidosis', 'ketones', 'ketonuria',
            'hyperglycemia', 'high glucose', 'elevated glucose', 'high blood sugar',
            'hypoglycemia', 'low glucose', 'low blood sugar', 'sweating', 'shakiness'
        }
        
        # Respiratory symptoms
        self.respiratory_symptoms = {
            'shortness of breath', 'dyspnea', 'breathlessness', 'difficulty breathing', 'sob',
            'severe shortness of breath', 'unable to breathe', 'respiratory distress',
            'gasping', 'choking', 'air hunger', 'severe dyspnea',
            'wheezing', 'wheeze', 'stridor', 'audible wheezing',
            'cough', 'coughing', 'persistent cough', 'productive cough', 'dry cough',
            'chest tightness', 'chest pressure', 'chest discomfort',
            'rapid breathing', 'tachypnea', 'fast breathing', 'increased respiratory rate',
            'cyanosis', 'blue lips', 'blue skin', 'bluish discoloration', 'hypoxia',
            'sputum', 'phlegm', 'mucus production',
            'pleuritic pain', 'pleurisy', 'chest pain with breathing'
        }
        
        # Infection-related symptoms
        self.infection_symptoms = {
            'fever', 'pyrexia', 'hyperthermia', 'elevated temperature', 'high fever',
            'fever > 103', 'fever > 39.4', 'fever > 38', 'low-grade fever',
            'chills', 'rigors', 'shaking chills', 'feeling cold',
            'sweating', 'night sweats', 'profuse sweating',
            'fatigue', 'weakness', 'malaise', 'generalized weakness',
            'headache', 'severe headache',
            'muscle aches', 'myalgia', 'body aches', 'generalized aches',
            'joint pain', 'arthralgia', 'joint stiffness',
            'nausea', 'vomiting', 'nausea and vomiting',
            'diarrhea', 'loose stools', 'watery stools',
            'redness', 'erythema', 'inflammation', 'swelling', 'edema',
            'warmth', 'warm to touch', 'localized warmth',
            'pus', 'purulent discharge', 'discharge', 'drainage',
            'lymph node swelling', 'swollen lymph nodes', 'lymphadenopathy',
            'sore throat', 'pharyngitis', 'throat pain',
            'runny nose', 'rhinorrhea', 'nasal congestion', 'stuffy nose'
        }
        
        # Combine all symptoms for quick lookup
        self.all_symptoms = (
            self.cardiac_symptoms |
            self.neurological_symptoms |
            self.diabetic_symptoms |
            self.respiratory_symptoms |
            self.infection_symptoms
        )

    def extract_entities(self, text: str) -> NERResult:
        """Extract medical entities from text with unified NER + regex + dictionary extraction."""
        entities = []
        
        # Step 1: NER model extraction
        if self.ner_pipeline:
            model_entities = self.ner_pipeline(text)
            for ent in model_entities:
                entity_type = self._map_entity_type(ent.get('entity_group', ''))
                if entity_type:
                    entities.append(ExtractedEntity(
                        text=ent['word'],
                        type=entity_type,
                        start=ent.get('start', 0),
                        end=ent.get('end', 0),
                        confidence=ent.get('score', 0.0)
                    ))
        
        # Step 2: Rule-based extraction (drugs, dosages, frequencies, durations)
        rule_entities = self._rule_based_extraction(text)
        entities.extend(rule_entities)
        
        # Step 3: Dictionary-based symptom extraction
        dict_symptoms = self._extract_symptoms_from_dictionaries(text)
        entities.extend(dict_symptoms)
        
        # Step 4: Enhanced regex-based vitals and lab extraction
        regex_vitals = self._extract_vitals_with_regex(text)
        entities.extend(regex_vitals)
        
        regex_labs = self._extract_labs_with_regex(text)
        entities.extend(regex_labs)
        
        # Step 5: Merge and deduplicate
        unified_entities = self._merge_and_deduplicate(entities, text)
        
        return NERResult(
            entities=unified_entities,
            raw_text=text,
            normalized_entities={}
        )
    
    def _extract_symptoms_from_dictionaries(self, text: str) -> List[ExtractedEntity]:
        """Extract symptoms using comprehensive dictionaries."""
        entities = []
        text_lower = text.lower()
        
        # Check all symptoms in dictionaries
        found_symptoms: Set[str] = set()
        
        for symptom in self.all_symptoms:
            # Create pattern that matches word boundaries
            pattern = r'\b' + re.escape(symptom) + r'\b'
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                symptom_text = match.group()
                # Avoid duplicates
                if symptom_text.lower() not in found_symptoms:
                    found_symptoms.add(symptom_text.lower())
                    entities.append(ExtractedEntity(
                        text=symptom_text,
                        type=EntityType.SYMPTOM,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.85
                    ))
        
        return entities
    
    def _extract_vitals_with_regex(self, text: str) -> List[ExtractedEntity]:
        """Extract vitals using enhanced regex patterns."""
        entities = []
        text_lower = text.lower()
        
        # Temperature patterns (Fahrenheit and Celsius)
        temp_patterns = [
            r'\b(temperature|temp|fever|t)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(°f|°F|fahrenheit|f)\b',
            r'\b(temperature|temp|fever|t)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(°c|°C|celsius|c)\b',
            r'\b(\d+(?:\.\d+)?)\s*(°f|°F|fahrenheit|f)\s*(temperature|temp|fever)?\b',
            r'\b(\d+(?:\.\d+)?)\s*(°c|°C|celsius|c)\s*(temperature|temp|fever)?\b',
            r'\b(fever|temp)\s+of\s+(\d+(?:\.\d+)?)\s*(degrees?)?\b'
        ]
        
        for pattern in temp_patterns:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    type=EntityType.VITALS,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9
                ))
        
        # Blood Pressure patterns
        bp_patterns = [
            r'\b(bp|blood pressure|blood pressure)\s*[:=]?\s*(\d+)\s*[/-]\s*(\d+)\s*(mmhg|mmHg|mm hg)?\b',
            r'\b(\d+)\s*[/-]\s*(\d+)\s*(mmhg|mmHg|mm hg)\b',
            r'\b(blood pressure|bp)\s+(\d+)\s*[/-]\s*(\d+)\b',
            r'\b(systolic|diastolic)\s+(\d+)\s*(mmhg|mmHg)?\b'
        ]
        
        for pattern in bp_patterns:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    type=EntityType.VITALS,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9
                ))
        
        # Glucose patterns
        glucose_patterns = [
            r'\b(glucose|blood sugar|bs|bg|blood glucose)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(mg/dl|mg/dL|mmol/l|mmol/L|mg|mmol)\b',
            r'\b(glucose|blood sugar)\s+of\s+(\d+(?:\.\d+)?)\s*(mg/dl|mg/dL|mmol/l|mmol/L)\b',
            r'\b(\d+(?:\.\d+)?)\s*(mg/dl|mg/dL|mmol/l|mmol/L)\s*(glucose|blood sugar)?\b'
        ]
        
        for pattern in glucose_patterns:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    type=EntityType.LAB_VALUE,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9
                ))
        
        # Oxygen saturation patterns
        oxygen_patterns = [
            r'\b(oxygen|o2|spo2|saturation|o2 sat|oxygen saturation)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(%|percent|percent)\b',
            r'\b(oxygen|o2|spo2)\s+(\d+(?:\.\d+)?)\s*(%|percent)\b',
            r'\b(\d+(?:\.\d+)?)\s*(%|percent)\s*(oxygen|o2|spo2|saturation)?\b',
            r'\b(sao2|spO2)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(%|percent)?\b'
        ]
        
        for pattern in oxygen_patterns:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    type=EntityType.VITALS,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9
                ))
        
        # Heart rate patterns
        hr_patterns = [
            r'\b(heart rate|hr|pulse|pulse rate)\s*[:=]?\s*(\d+)\s*(bpm|beats?/min|beats? per minute|beats?/minute)?\b',
            r'\b(heart rate|hr|pulse)\s+of\s+(\d+)\s*(bpm|beats?/min)?\b',
            r'\b(\d+)\s*(bpm|beats?/min|beats? per minute)\s*(heart rate|hr|pulse)?\b',
            r'\b(pulse)\s+(\d+)\s*(bpm)?\b'
        ]
        
        for pattern in hr_patterns:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    type=EntityType.VITALS,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9
                ))
        
        # Respiratory rate patterns
        rr_patterns = [
            r'\b(respiratory rate|rr|resp rate|breathing rate)\s*[:=]?\s*(\d+)\s*(breaths?/min|breaths? per minute|breaths?/minute)?\b',
            r'\b(respiratory rate|rr)\s+of\s+(\d+)\s*(breaths?/min)?\b',
            r'\b(\d+)\s*(breaths?/min|breaths? per minute)\s*(respiratory rate|rr)?\b'
        ]
        
        for pattern in rr_patterns:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    type=EntityType.VITALS,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9
                ))
        
        return entities
    
    def _extract_labs_with_regex(self, text: str) -> List[ExtractedEntity]:
        """Extract lab values using enhanced regex patterns."""
        entities = []
        text_lower = text.lower()
        
        # Comprehensive lab patterns
        lab_patterns = [
            # Creatinine
            r'\b(creatinine|creat|cr)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(mg/dl|mg/dL|μmol/l|μmol/L|mg|μmol)\b',
            # Hemoglobin
            r'\b(hemoglobin|hgb|hb)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(g/dl|g/dL|g/l|g/L|g)\b',
            # HbA1c
            r'\b(hba1c|hba1c|a1c|glycated hemoglobin|hba1c)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(%|percent)\b',
            # Cholesterol
            r'\b(cholesterol|chol|total cholesterol)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(mg/dl|mg/dL|mmol/l|mmol/L)\b',
            # LDL
            r'\b(ldl|low density lipoprotein)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(mg/dl|mg/dL|mmol/l|mmol/L)\b',
            # HDL
            r'\b(hdl|high density lipoprotein)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(mg/dl|mg/dL|mmol/l|mmol/L)\b',
            # Triglycerides
            r'\b(triglycerides|tg|trig)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(mg/dl|mg/dL|mmol/l|mmol/L)\b',
            # ALT
            r'\b(alt|alanine aminotransferase|sgot)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(units?/l|u/l|iu/l)\b',
            # AST
            r'\b(ast|aspartate aminotransferase|sgpt)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(units?/l|u/l|iu/l)\b',
            # TSH
            r'\b(tsh|thyroid stimulating hormone)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(miu/l|mIU/L|μIU/ml)\b',
            # T3
            r'\b(t3|triiodothyronine)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(ng/dl|ng/dL|pmol/l|pmol/L)\b',
            # T4
            r'\b(t4|thyroxine)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(μg/dl|μg/dL|nmol/l|nmol/L)\b',
            # INR
            r'\b(inr|international normalized ratio)\s*[:=]?\s*(\d+(?:\.\d+)?)\b',
            # PT
            r'\b(pt|prothrombin time)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(seconds?|sec)\b',
            # eGFR
            r'\b(egfr|estimated glomerular filtration rate|gfr)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(ml/min|mL/min)\b'
        ]
        
        for pattern in lab_patterns:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    type=EntityType.LAB_VALUE,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85
                ))
        
        return entities
    
    def _merge_and_deduplicate(self, entities: List[ExtractedEntity], text: str) -> List[ExtractedEntity]:
        """Merge NER, regex, and dictionary results, removing duplicates."""
        # Use position-based deduplication
        seen_positions: Set[tuple] = set()
        unified: List[ExtractedEntity] = []
        
        # Sort by start position
        entities_sorted = sorted(entities, key=lambda e: (e.start, e.end))
        
        for entity in entities_sorted:
            # Create position key
            pos_key = (entity.start, entity.end, entity.type)
            
            # Check for overlap with existing entities
            overlaps = False
            for existing in unified:
                if (entity.start <= existing.end and entity.end >= existing.start and 
                    entity.type == existing.type):
                    # Overlapping entities of same type - keep the longer one
                    if len(entity.text) > len(existing.text):
                        unified.remove(existing)
                        unified.append(entity)
                    overlaps = True
                    break
            
            if not overlaps and pos_key not in seen_positions:
                seen_positions.add(pos_key)
                unified.append(entity)
        
        # Final deduplication by text content and type
        final_entities = []
        seen_text_type: Set[tuple] = set()
        
        for entity in unified:
            text_lower = entity.text.lower().strip()
            key = (text_lower, entity.type)
            if key not in seen_text_type:
                seen_text_type.add(key)
                final_entities.append(entity)
        
        return final_entities

    def _map_entity_type(self, model_label: str) -> EntityType:
        """Map model labels to our entity types."""
        return None  # Rely on rule-based extraction

    def _rule_based_extraction(self, text: str) -> List[ExtractedEntity]:
        """Enhanced rule-based extraction for drugs, dosages, frequencies, durations."""
        entities = []
        text_lower = text.lower()
        
        # Enhanced Drug patterns
        drug_patterns = [
            r'\b(aspirin|ibuprofen|paracetamol|acetaminophen|metformin|insulin|warfarin|amoxicillin|penicillin|atorvastatin|simvastatin|omeprazole|lisinopril|amlodipine|metoprolol|furosemide|levothyroxine|gabapentin|sertraline|amiodarone|digoxin|morphine|oxycodone|hydrocodone|tramadol|codeine|diazepam|lorazepam|albuterol|prednisone|hydrochlorothiazide|spironolactone|carvedilol|propranolol|atenolol|losartan|valsartan|enalapril|captopril|ramipril|perindopril|trandolapril|quinapril|moexipril|benazepril|fosinopril|zestril|prinivil|cozaar|diovan|micardis|avapro|teveten|atacand|bystolic|coreg|toprol|lopressor|inderal|tenormin|zebeta|sectral|kerlone|betapace|betapace|betaxolol|cartrol|levatol|visken|normodyne|trandate|corgard|nadolol|timolol|blocadren|betimol|istalol|timoptic|optipranolol|betagan|betoptic|betimol|istalol|timoptic|optipranolol|betagan|betoptic)\b',
            r'\b([A-Z][a-z]+(?:pril|statin|mycin|azole|olol|sartan|pam|lam|ide|ine|ate|one|ium|tin|cin|micin|cycline|floxacin|oxacin|zepam|zolam|dipine|pril|statin|mycin|azole|olol|sartan))\b'
        ]
        
        for pattern in drug_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    type=EntityType.DRUG,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8
                ))
        
        # Enhanced Dosage patterns
        dosage_patterns = [
            r'\b(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg|units?|tablets?|capsules?|pills?|drops?)\b',
            r'\b(\d+/\d+)\s*(mg|g|ml|tablet|cap|pill)\b',
            r'\b(one|two|three|four|five|half|quarter|½|¼|¾)\s*(tablet|cap|pill|mg|g)\b',
            r'\b(\d+(?:\.\d+)?)\s*(milligrams?|grams?|milliliters?|micrograms?)\b'
        ]
        
        for pattern in dosage_patterns:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    type=EntityType.DOSAGE,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85
                ))
        
        # Enhanced Frequency patterns
        freq_patterns = [
            r'\b(qd|bid|tid|qid|qod|prn|once|twice|thrice|daily|weekly|monthly|as needed)\b',
            r'\b(\d+)\s*(times?|x)\s*(per|a|daily|week|day|hour)\b',
            r'\b(every\s+\d+\s*(hours?|days?|weeks?))\b'
        ]
        
        for pattern in freq_patterns:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    type=EntityType.FREQUENCY,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85
                ))
        
        # Enhanced Duration patterns
        duration_patterns = [
            r'\b(\d+)\s*(days?|weeks?|months?|years?)\b',
            r'\b(for|duration|course of)\s+(\d+)\s*(days?|weeks?|months?)\b',
            r'\b(\d+)\s*-\s*(\d+)\s*(days?|weeks?)\b'
        ]
        
        for pattern in duration_patterns:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    type=EntityType.DURATION,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8
                ))
        
        return entities
