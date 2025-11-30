from typing import List, Dict, Any
from app.models.schemas import AgentOutput, Severity, AuditReport
from app.services.next_steps_generator import NextStepsGenerator
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

# Severity order and base scores
SEVERITY_ORDER = ["ok", "low", "moderate", "high", "critical"]
SEVERITY_BASE_SCORE = {
    "ok": 0.0,
    "low": 1.0,        # 1-2 mapped by agents
    "moderate": 4.0,   # 3-5, choose a default center
    "high": 7.0,       # 6-8
    "critical": 10.0,  # 9-10 -> use 10 as base
}

SEVERITY_RANGES = {
    "ok": (0.0, 0.0),
    "low": (1.0, 2.0),
    "moderate": (3.0, 5.0),
    "high": (6.0, 8.0),
    "critical": (9.0, 10.0)
}


class DecisionEngine:
    """Decision engine that merges all agent outputs into final audit report."""
    
    @staticmethod
    def clamp_score_by_severity(score: float, severity: str) -> float:
        """Clamp score into range per severity."""
        lo, hi = SEVERITY_RANGES.get(severity.lower(), (0.0, 10.0))
        if score < lo:
            return lo
        if score > hi:
            return hi
        return score
    
    @staticmethod
    def normalize_agent(agent_output: AgentOutput) -> Dict[str, Any]:
        """Normalize agent output to ensure consistent format."""
        status_str = agent_output.status.value if isinstance(agent_output.status, Severity) else str(agent_output.status)
        score = float(agent_output.score) if agent_output.score is not None else SEVERITY_BASE_SCORE.get(status_str, 0.0)
        
        # Clamp score to severity range
        score = DecisionEngine.clamp_score_by_severity(score, status_str)
        
        return {
            "agent": agent_output.agent,
            "status": status_str,
            "score": score,
            "message": agent_output.message,
            "evidence": agent_output.evidence,
            "details": agent_output.details
        }
    
    @staticmethod
    def merge_agent_outputs(agent_outputs: List[AgentOutput], ner_result: Any = None, 
                           pattern_detections: List[Dict] = None) -> AuditReport:
        """
        Merge all agent outputs into final audit report.
        
        Policy:
        - Pick the highest severity among agents (ok < low < moderate < high < critical)
        - For the highest severity, use the maximum agent score within that severity
        - Clamp final score to severity range
        - If no agents present → score = 0
        """
        logger.info(f"Merging {len(agent_outputs)} agent outputs")
        
        # Log inputs
        logger.debug(f"Agent outputs: {[{'agent': a.agent, 'status': a.status.value if isinstance(a.status, Severity) else str(a.status), 'score': a.score} for a in agent_outputs]}")
        
        if not agent_outputs:
            logger.warning("No agent outputs provided, returning safe default")
            return DecisionEngine._create_empty_report(ner_result)
        
        # Normalize all agents
        normalized = [DecisionEngine.normalize_agent(a) for a in agent_outputs]
        logger.debug(f"Normalized agents: {normalized}")
        
        # Determine highest severity present
        highest_idx = 0
        highest_sev = "ok"
        all_agents_ok = True
        all_agents_ok_or_low = True
        
        for agent in normalized:
            status_str = agent["status"].lower()
            if status_str != "ok":
                all_agents_ok = False
            if status_str not in ["ok", "low"]:
                all_agents_ok_or_low = False
            if status_str in SEVERITY_ORDER:
                idx = SEVERITY_ORDER.index(status_str)
                if idx > highest_idx:
                    highest_idx = idx
                    highest_sev = status_str
        
        logger.info(f"Highest severity detected: {highest_sev}")
        
        # If ALL agents are OK, force final score to 0 (SAFE) regardless of symptoms
        if all_agents_ok:
            logger.info("=" * 60)
            logger.info("ALL AGENTS OK - FORCING SCORE TO 0 (SAFE)")
            logger.info("  - All agents returned status='ok'")
            logger.info("  - Final score will be 0.0 regardless of symptom presence")
            logger.info("=" * 60)
            highest_sev = "ok"
        # If all agents are OK or LOW, and highest is LOW, keep it as LOW (don't escalate)
        elif all_agents_ok_or_low and highest_sev == "low":
            logger.info("All agents OK or LOW - keeping severity as LOW (1-2)")
            highest_sev = "low"
        
        # Among agents with that severity, pick max reported score
        relevant_agents = [a for a in normalized if a["status"].lower() == highest_sev]
        relevant_scores = [a["score"] for a in relevant_agents]
        
        if not relevant_scores:
            # Fallback to base mapping of highest severity
            final_score = float(SEVERITY_BASE_SCORE.get(highest_sev, 0.0))
            logger.warning(f"No scores found for severity {highest_sev}, using base score {final_score}")
        else:
            # Use max score among agents with highest severity
            final_score = float(max(relevant_scores))
            logger.info(f"Max score among {highest_sev} agents: {final_score}")
        
        # If all agents are OK, force score to 0 (SAFE)
        if all_agents_ok:
            final_score = 0.0
            logger.info(f"Forced final score to 0.0 because all agents are OK")
        # If all agents are OK or LOW, ensure score stays in LOW range (1-2)
        elif all_agents_ok_or_low and highest_sev == "low":
            # Use the max LOW score from agents, but clamp to 1-2 range
            low_agents = [a for a in normalized if a["status"].lower() == "low"]
            if low_agents:
                low_scores = [a["score"] for a in low_agents]
                final_score = min(2.0, max(1.0, max(low_scores)))
                logger.info(f"All agents OK or LOW - clamped score to LOW range: {final_score}")
        # If all agents are OK or LOW, ensure score stays in LOW range (1-2)
        elif all_agents_ok_or_low and highest_sev == "low":
            # Use the max LOW score from agents, but clamp to 1-2 range
            low_agents = [a for a in normalized if a["status"].lower() == "low"]
            if low_agents:
                low_scores = [a["score"] for a in low_agents]
                final_score = min(2.0, max(1.0, max(low_scores)))
                logger.info(f"All agents OK or LOW - clamped score to LOW range: {final_score}")
        
        # Clamp to severity range
        final_score = DecisionEngine.clamp_score_by_severity(final_score, highest_sev)
        logger.info(f"Final clamped score: {final_score} (severity: {highest_sev})")
        
        # Check for pattern detections (override agent scoring)
        if pattern_detections:
            for detection in pattern_detections:
                if detection.get('override_agent_score', False):
                    pattern_score = detection.get('score', 10.0)
                    pattern_score = DecisionEngine.clamp_score_by_severity(pattern_score, 'critical')
                    if pattern_score > final_score:
                        final_score = pattern_score
                        highest_sev = 'critical'
                        logger.info(f"Pattern detection overrode score to {final_score}")
        
        # Categorize issues by severity
        critical_issues = []
        high_issues = []
        moderate_issues = []
        low_issues = []
        safe_items = []
        
        for agent in normalized:
            status = agent["status"].lower()
            score = agent["score"]
            
            issue_data = {
                'agent': agent["agent"],
                'message': agent["message"],
                'evidence': agent["evidence"],
                'details': agent["details"],
                'score': score,
                'confidence': DecisionEngine._calculate_confidence_from_agent(agent),
                'confidence_explanation': DecisionEngine._generate_confidence_explanation_from_agent(agent),
            }
            
            if status == "critical":
                issue_data['why_risky'] = DecisionEngine._generate_risk_explanation_from_agent(agent)
                critical_issues.append(issue_data)
            elif status == "high":
                issue_data['why_risky'] = DecisionEngine._generate_risk_explanation_from_agent(agent)
                high_issues.append(issue_data)
            elif status == "moderate":
                moderate_issues.append(issue_data)
            elif status == "low":
                low_issues.append(issue_data)
            else:  # ok
                safe_items.append(issue_data)
        
        # NORMAL PATIENT OVERRIDE - Takes priority over all other scoring
        # This override ensures truly normal patients get score 0 regardless of agent outputs
        override_applied = False
        if ner_result and hasattr(ner_result, 'normalized_entities'):
            normalized_entities = ner_result.normalized_entities
            
            # Check conditions for normal patient override
            symptoms = normalized_entities.get('symptoms', [])
            vitals = normalized_entities.get('vitals', [])
            lab_values = normalized_entities.get('lab_values', [])
            drugs = normalized_entities.get('drugs', [])
            
            # Condition 1: No symptoms detected
            has_no_symptoms = len(symptoms) == 0
            
            # Condition 2: All vitals in normal ranges
            vitals_normal = True
            if vitals:
                for vital in vitals:
                    if isinstance(vital, dict):
                        vital_type = str(vital.get('type', '')).lower()
                        vital_value = vital.get('value')
                        vital_text = str(vital.get('text', '')).lower()
                        
                        # BP: 90-130 / 60-85
                        if 'blood pressure' in vital_type or 'bp' in vital_type or 'blood pressure' in vital_text:
                            if vital_value:
                                if isinstance(vital_value, (list, tuple)) and len(vital_value) >= 2:
                                    systolic, diastolic = vital_value[0], vital_value[1]
                                    if not (90 <= systolic <= 130 and 60 <= diastolic <= 85):
                                        vitals_normal = False
                                        break
                        # Pulse/HR: 60-100
                        elif 'heart rate' in vital_type or 'hr' in vital_type or 'pulse' in vital_type or 'heart rate' in vital_text or 'bpm' in vital_text:
                            if vital_value:
                                hr = float(vital_value) if vital_value else None
                                if hr and not (60 <= hr <= 100):
                                    vitals_normal = False
                                    break
                        # Oxygen: >= 95%
                        elif 'oxygen' in vital_type or 'spo2' in vital_type or 'oxygen' in vital_text or 'spo2' in vital_text:
                            if vital_value:
                                o2 = float(vital_value) if vital_value else None
                                if o2 and o2 < 95:
                                    vitals_normal = False
                                    break
                        # Temperature: 97-99.5 F
                        elif 'temperature' in vital_type or 'temp' in vital_type or 'fever' in vital_type or 'temperature' in vital_text or '°f' in vital_text:
                            if vital_value:
                                temp = float(vital_value) if vital_value else None
                                if temp and not (97 <= temp <= 99.5):
                                    vitals_normal = False
                                    break
                    else:
                        # String format - try to extract values
                        vital_str = str(vital).lower()
                        import re
                        # Check for BP pattern
                        bp_match = re.search(r'(\d+)\s*/\s*(\d+)', vital_str)
                        if bp_match:
                            sys, dia = int(bp_match.group(1)), int(bp_match.group(2))
                            if not (90 <= sys <= 130 and 60 <= dia <= 85):
                                vitals_normal = False
                                break
                        # Check for HR pattern
                        hr_match = re.search(r'(\d+)\s*(?:bpm|heart rate|hr)', vital_str)
                        if hr_match:
                            hr = int(hr_match.group(1))
                            if not (60 <= hr <= 100):
                                vitals_normal = False
                                break
                        # Check for O2 pattern
                        o2_match = re.search(r'(\d+(?:\.\d+)?)\s*%', vital_str)
                        if o2_match and ('oxygen' in vital_str or 'spo2' in vital_str):
                            o2 = float(o2_match.group(1))
                            if o2 < 95:
                                vitals_normal = False
                                break
                        # Check for temp pattern
                        temp_match = re.search(r'(\d+(?:\.\d+)?)\s*°?f', vital_str)
                        if temp_match:
                            temp = float(temp_match.group(1))
                            if not (97 <= temp <= 99.5):
                                vitals_normal = False
                                break
            
            # Condition 3: All labs normal (glucose 70-140 mg/dL)
            labs_normal = True
            if lab_values:
                for lab in lab_values:
                    if isinstance(lab, dict):
                        lab_type = str(lab.get('type', '')).lower()
                        lab_value = lab.get('value')
                        lab_text = str(lab.get('text', '')).lower()
                        
                        # Glucose: 70-140 mg/dL
                        if 'glucose' in lab_type or 'blood sugar' in lab_type or 'glucose' in lab_text:
                            if lab_value:
                                glucose = float(lab_value) if lab_value else None
                                if glucose and not (70 <= glucose <= 140):
                                    labs_normal = False
                                    break
                    else:
                        # String format
                        lab_str = str(lab).lower()
                        import re
                        if 'glucose' in lab_str or 'blood sugar' in lab_str:
                            glucose_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:mg/dl|mg/dL)', lab_str)
                            if glucose_match:
                                glucose = float(glucose_match.group(1))
                                if not (70 <= glucose <= 140):
                                    labs_normal = False
                                    break
            
            # Apply override if vitals/labs/symptoms are all normal
            # This takes priority over agent scores (even if agents report high/critical)
            if has_no_symptoms and vitals_normal and labs_normal:
                logger.info("=" * 60)
                logger.info("NORMAL PATIENT OVERRIDE APPLIED")
                logger.info(f"  - No symptoms: {has_no_symptoms}")
                logger.info(f"  - Vitals normal: {vitals_normal}")
                logger.info(f"  - Labs normal: {labs_normal}")
                logger.info(f"  - OVERRIDING: severity=ok, score=0 (was: {highest_sev}, {final_score})")
                logger.info(f"  - NOTE: Override takes priority over agent scores (even if agents report high/critical)")
                logger.info("=" * 60)
                
                # Override final output
                final_score = 0.0
                highest_sev = "ok"
                override_applied = True
                
                # Clear all issues and set safe
                critical_issues = []
                high_issues = []
                moderate_issues = []
                low_issues = []
                safe_items = [{
                    'agent': 'System',
                    'message': 'No concerning findings. Routine health maintenance.',
                    'score': 0.0,
                    'confidence': 1.0,
                    'confidence_explanation': 'High confidence: All vitals, labs, and symptoms within normal ranges.',
                    'evidence': [],
                    'details': {}
                }]
        
        # Generate recommendations
        if override_applied:
            # Use simple recommendation for normal patients
            recommendations = ["No concerning findings. Routine health maintenance."]
        else:
            recommendations = DecisionEngine._generate_recommendations(
                critical_issues, high_issues, moderate_issues, low_issues, agent_outputs
            )
        
        # Generate structured next steps
        next_steps_generator = NextStepsGenerator()
        next_steps = next_steps_generator.generate_next_steps(
            critical_issues=critical_issues,
            high_issues=high_issues,
            moderate_issues=moderate_issues,
            low_issues=low_issues,
            agent_outputs=agent_outputs,
            ner_result=ner_result,
            risk_score=final_score
        )
        
        # Log next steps generation
        logger.info(f"Generated next steps: urgency={next_steps.urgency_level}, items={len(next_steps.items)}")
        for item in next_steps.items:
            logger.info(f"  - {item.title} ({item.priority}) by {item.recommended_by_agent}, evidence_ids={item.evidence_ids}")
        
        # Add low issues to moderate for backward compatibility
        moderate_issues.extend(low_issues)
        
        # Create audit report
        audit_id = str(uuid.uuid4())
        
        logger.info(f"Final audit report: severity={highest_sev}, score={final_score:.2f}, "
                   f"critical={len(critical_issues)}, high={len(high_issues)}, "
                   f"moderate={len(moderate_issues)}, safe={len(safe_items)}")
        
        return AuditReport(
            audit_id=audit_id,
            timestamp=datetime.utcnow(),
            critical_issues=critical_issues,
            high_issues=high_issues,
            moderate_issues=moderate_issues,
            safe_items=safe_items,
            recommendations=recommendations,
            risk_score=final_score,
            agent_outputs=agent_outputs,
            ner_result=ner_result,
            next_steps=next_steps
        )
    
    @staticmethod
    def _create_empty_report(ner_result: Any = None) -> AuditReport:
        """Create an empty safe report."""
        return AuditReport(
            audit_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            critical_issues=[],
            high_issues=[],
            moderate_issues=[],
            safe_items=[],
            recommendations=["No issues detected. Patient appears stable."],
            risk_score=0.0,
            agent_outputs=[],
            ner_result=ner_result
        )
    
    @staticmethod
    def _calculate_confidence_from_agent(agent: Dict[str, Any]) -> float:
        """Calculate confidence level (0-1) based on evidence and agent type."""
        base_confidence = 0.7
        
        # Increase confidence with more evidence
        evidence_count = len(agent.get('evidence', []))
        evidence_boost = min(0.2, evidence_count * 0.05)
        
        # Agent-specific confidence adjustments
        agent_confidence = {
            'DosageChecker': 0.85,
            'InteractionChecker': 0.80,
            'RedFlagChecker': 0.90,
            'MissingTestsChecker': 0.75,
            'GuidelineComplianceChecker': 0.70,
            'PatternDetector': 0.95
        }
        
        agent_base = agent_confidence.get(agent.get('agent', ''), 0.7)
        confidence = min(1.0, agent_base + evidence_boost)
        
        # Adjust based on severity (higher severity = higher confidence)
        status = agent.get('status', 'ok').lower()
        if status == 'critical':
            confidence = min(1.0, confidence + 0.05)
        elif status == 'high':
            confidence = min(1.0, confidence + 0.03)
        
        return round(confidence, 2)
    
    @staticmethod
    def _generate_confidence_explanation_from_agent(agent: Dict[str, Any]) -> str:
        """Generate explanation of confidence level."""
        confidence = DecisionEngine._calculate_confidence_from_agent(agent)
        evidence_count = len(agent.get('evidence', []))
        
        if confidence >= 0.9:
            level = "Very High"
            explanation = f"Very high confidence ({confidence*100:.0f}%) based on {evidence_count} evidence sources and established medical patterns."
        elif confidence >= 0.8:
            level = "High"
            explanation = f"High confidence ({confidence*100:.0f}%) based on {evidence_count} evidence sources and clinical guidelines."
        elif confidence >= 0.7:
            level = "Moderate-High"
            explanation = f"Moderate-high confidence ({confidence*100:.0f}%) based on {evidence_count} evidence sources. Clinical review recommended."
        elif confidence >= 0.6:
            level = "Moderate"
            explanation = f"Moderate confidence ({confidence*100:.0f}%) based on {evidence_count} evidence sources. Additional clinical evaluation may be needed."
        else:
            level = "Low-Moderate"
            explanation = f"Low-moderate confidence ({confidence*100:.0f}%) based on limited evidence. Clinical judgment required."
        
        return f"{level} Confidence: {explanation}"
    
    @staticmethod
    def _generate_risk_explanation_from_agent(agent: Dict[str, Any]) -> str:
        """Generate explanation of why this is risky."""
        agent_name = agent.get('agent', '')
        message = agent.get('message', '')
        
        if agent_name == 'DosageChecker':
            return "Overdose or underdose can cause serious adverse effects, treatment failure, or toxicity. Proper dosing is critical for medication safety and efficacy."
        elif agent_name == 'InteractionChecker':
            return "Drug interactions can lead to increased toxicity, reduced efficacy, or life-threatening adverse reactions. Some combinations are contraindicated."
        elif agent_name == 'RedFlagChecker':
            return "Red flag symptoms indicate potential medical emergencies that require immediate attention. Delayed treatment can lead to serious complications or death."
        elif agent_name == 'MissingTestsChecker':
            return "Missing essential tests can lead to undetected adverse effects, organ damage, or inappropriate dosing. Baseline monitoring is crucial for safe medication use."
        elif agent_name == 'GuidelineComplianceChecker':
            return "Non-compliance with treatment guidelines may result in suboptimal outcomes, increased complications, or avoidable adverse events."
        else:
            return "This finding requires attention to ensure patient safety and optimal treatment outcomes."

    @staticmethod
    def _generate_recommendations(
        critical_issues: List[Dict],
        high_issues: List[Dict],
        moderate_issues: List[Dict],
        low_issues: List[Dict],
        agent_outputs: List[AgentOutput]
    ) -> List[str]:
        """Generate actionable recommendations with evidence."""
        recommendations = []
        
        # Generate severity-specific recommendations with clinically appropriate language
        if critical_issues:
            avg_score = sum(iss.get('score', 10.0) for iss in critical_issues) / len(critical_issues) if critical_issues else 10.0
            severity_explanation = f"CRITICAL FINDINGS (Risk Score: {avg_score:.1f}/10): Life-threatening conditions detected. Immediate emergency department evaluation and intervention required."
        elif high_issues:
            avg_score = sum(iss.get('score', 7.0) for iss in high_issues) / len(high_issues) if high_issues else 7.0
            severity_explanation = f"HIGH PRIORITY FINDINGS (Risk Score: {avg_score:.1f}/10): Serious conditions identified requiring urgent clinical evaluation within 24-48 hours."
        elif moderate_issues:
            avg_score = sum(iss.get('score', 4.0) for iss in moderate_issues) / len(moderate_issues) if moderate_issues else 4.0
            severity_explanation = f"MODERATE CONCERNS (Risk Score: {avg_score:.1f}/10): Clinical evaluation recommended to assess and monitor patient status."
        elif low_issues:
            avg_score = sum(iss.get('score', 1.5) for iss in low_issues) / len(low_issues) if low_issues else 1.5
            severity_explanation = f"MINOR FINDINGS (Risk Score: {avg_score:.1f}/10): Mild symptoms present. Continue monitoring and consider evaluation if symptoms persist or worsen."
        else:
            severity_explanation = "NO SIGNIFICANT FINDINGS (Risk Score: 0/10): No critical issues detected. Patient appears stable based on current assessment."
        
        recommendations.append(severity_explanation)
        
        # Critical issues - use clinically appropriate language
        for issue in critical_issues:
            agent = issue.get('agent', '')
            evidence = issue.get('evidence', [])
            score = issue.get('score', 10.0)
            message = issue.get('message', '')
            
            if agent == 'DosageChecker':
                rec = f"CRITICAL: Medication dosing issue detected (Risk Score: {score:.1f}/10). Immediate dose adjustment required. Consult clinical pharmacist or prescribing physician for dose modification. Monitor for adverse effects."
                if evidence:
                    rec += f" Clinical evidence: {evidence[0][:150]}..." if evidence else ""
                recommendations.append(rec)
            elif agent == 'InteractionChecker':
                rec = f"CRITICAL: Significant drug-drug interaction identified (Risk Score: {score:.1f}/10). Consider discontinuing one medication or switching to alternative therapy. Monitor patient for signs of interaction-related adverse effects."
                if evidence:
                    rec += f" Interaction details: {evidence[0][:150]}..." if evidence else ""
                recommendations.append(rec)
            elif agent == 'RedFlagChecker' or agent == 'PatternDetector':
                rec = f"CRITICAL: Emergency red flag symptoms detected (Risk Score: {score:.1f}/10). Immediate emergency department evaluation recommended. {message[:200] if message else 'Patient requires urgent medical assessment.'}"
                if evidence:
                    rec += f" Clinical indicators: {evidence[0][:150]}..." if evidence else ""
                recommendations.append(rec)
        
        # High issues - use professional medical language
        for issue in high_issues:
            agent = issue.get('agent', '')
            evidence = issue.get('evidence', [])
            score = issue.get('score', 7.0)
            message = issue.get('message', '')
            
            if agent == 'MissingTestsChecker':
                rec = f"HIGH PRIORITY: Essential laboratory monitoring required (Risk Score: {score:.1f}/10). Order baseline and follow-up labs as indicated before continuing current medication regimen. Baseline monitoring is critical for patient safety."
                if evidence:
                    rec += f" Recommended tests: {evidence[0][:150]}..." if evidence else ""
                recommendations.append(rec)
            elif agent == 'RedFlagChecker':
                # Extract key clinical findings from message
                rec = f"HIGH PRIORITY: Clinical findings require urgent evaluation (Risk Score: {score:.1f}/10). {message[:300] if message else 'Patient requires urgent clinical assessment.'}"
                recommendations.append(rec)
            else:
                rec = f"HIGH PRIORITY: {message[:300] if message else 'Clinical evaluation recommended within 24-48 hours.'} (Risk Score: {score:.1f}/10)"
                recommendations.append(rec)
        
        # Moderate issues
        if moderate_issues:
            avg_score = sum(m.get('score', 4.0) for m in moderate_issues) / len(moderate_issues)
            rec = f"MODERATE PRIORITY: Clinical evaluation recommended (Risk Score: {avg_score:.1f}/10). Review treatment plan for guideline compliance and consider alternative therapeutic approaches if indicated."
            recommendations.append(rec)
        
        # Low issues
        if low_issues:
            avg_score = sum(l.get('score', 1.5) for l in low_issues) / len(low_issues)
            rec = f"MINOR FINDINGS: Continue monitoring (Risk Score: {avg_score:.1f}/10). Monitor symptoms and consider clinical evaluation if symptoms persist or worsen."
            recommendations.append(rec)
        
        # General recommendations - clinically appropriate
        if not critical_issues and not high_issues:
            recommendations.append("No critical or high-priority findings identified. Continue standard clinical monitoring and follow-up as per established protocols.")
        else:
            recommendations.append(
                "RECOMMENDED ACTIONS: Schedule follow-up appointment to assess patient response, monitor for adverse effects, and adjust treatment plan as clinically indicated."
            )
        
        return recommendations
