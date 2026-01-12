"""
Report viewer component for Streamlit dashboard.
"""
import streamlit as st
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import io


def display_ner_results(ner_result: Dict[str, Any]):
    """Display NER extraction results."""
    st.subheader("🔍 Extracted Entities")
    
    if not ner_result or not ner_result.get('entities'):
        st.info("No entities extracted")
        return
    
    entities = ner_result.get('entities', [])
    
    # Group by type
    entity_types = {}
    for entity in entities:
        entity_type = entity.get('type', 'UNKNOWN')
        if entity_type not in entity_types:
            entity_types[entity_type] = []
        entity_types[entity_type].append(entity)
    
    # Display in columns
    cols = st.columns(min(len(entity_types), 4))
    for idx, (entity_type, entity_list) in enumerate(entity_types.items()):
        with cols[idx % len(cols)]:
            st.markdown(f"**{entity_type}**")
            for entity in entity_list[:10]:  # Limit to 10 per type
                st.text(f"• {entity.get('text', '')}")
            if len(entity_list) > 10:
                st.caption(f"... and {len(entity_list) - 10} more")


def display_agent_outputs(agent_outputs: List[Dict[str, Any]]):
    """Display agent analysis outputs with RAG evidence viewer and LLM explanation chains."""
    st.subheader("🤖 Agent Analysis")
    
    for agent_output in agent_outputs:
        agent_name = agent_output.get('agent', 'Unknown')
        status = agent_output.get('status', 'ok')
        message = agent_output.get('message', '')
        score = agent_output.get('score', 0.0)
        
        # Status badge color
        status_colors = {
            'critical': '🔴',
            'high': '🟠',
            'moderate': '🟡',
            'low': '🔵',
            'ok': '🟢'
        }
        
        with st.expander(f"{status_colors.get(status.lower(), '⚪')} {agent_name} - {status.upper()}"):
            st.write(f"**Message:** {message}")
            st.write(f"**Score:** {score:.2f}/10")
            
            # LLM Explanation Chain
            details = agent_output.get('details', {})
            llm_explanation = details.get('llm_explanation', {}) if details else {}
            if llm_explanation:
                st.markdown("---")
                st.markdown("### 🧠 LLM Explanation Chain")
                
                why_flagged = llm_explanation.get('why_flagged', '')
                if why_flagged:
                    st.markdown(f"**Why this was flagged:** {why_flagged}")
                
                triggered_items = llm_explanation.get('triggered_items', [])
                if triggered_items:
                    st.markdown(f"**Triggered items:** {', '.join([str(item) for item in triggered_items[:5]])}")
                
                guideline_chunks = llm_explanation.get('guideline_chunks', [])
                if guideline_chunks:
                    st.markdown(f"**Retrieved guideline chunks:** {len(guideline_chunks)} chunks")
                    with st.expander("View Guideline Chunks"):
                        for chunk in guideline_chunks[:3]:
                            st.text(str(chunk)[:300])
                
                confidence = llm_explanation.get('confidence', 'medium')
                confidence_reasoning = llm_explanation.get('confidence_reasoning', '')
                st.markdown(f"**Confidence:** {confidence.upper()}")
                if confidence_reasoning:
                    st.caption(confidence_reasoning)
            
            # RAG Evidence Viewer - Enhanced
            evidence = agent_output.get('evidence', [])
            if evidence:
                st.markdown("---")
                st.markdown("### 📚 RAG Evidence (Retrieved Knowledge)")
                for idx, ev in enumerate(evidence[:5], 1):  # Show first 5
                    with st.expander(f"Evidence {idx}: View Retrieved Knowledge", expanded=False):
                        if isinstance(ev, str):
                            # Try to parse if it's a JSON string
                            try:
                                ev_dict = eval(ev) if ev.startswith('{') else None
                                if ev_dict:
                                    st.json(ev_dict)
                                else:
                                    st.text(ev)
                            except:
                                st.text(ev)
                        else:
                            st.json(ev) if isinstance(ev, dict) else st.text(str(ev))
            
            # Details section
            if details:
                st.markdown("---")
                st.markdown("### 🔍 Detailed Analysis")
                st.json(details)


def display_final_report(audit_report: Dict[str, Any]):
    """Display final medical audit report in professional, patient-friendly format (Mayo Clinic/NHS style)."""
    
    # ============================================
    # SECTION 1: FINAL RESULT SUMMARY
    # ============================================
    st.markdown("---")
    st.markdown("## 📋 Your Medical Assessment Report")
    
    risk_score = audit_report.get('risk_score', 0.0)
    ner_result = audit_report.get('ner_result', {})
    normalized = ner_result.get('normalized_entities', {}) if ner_result else {}
    
    # Determine risk level and urgency
    if risk_score == 0.0:
        risk_level = "SAFE"
        urgency = "Routine follow-up"
        urgency_icon = "🟢"
        risk_color = "green"
        summary_reason = "No concerning findings detected. Your symptoms and vital signs are within normal ranges."
    elif risk_score >= 9:
        risk_level = "CRITICAL"
        urgency = "Seek emergency care immediately"
        urgency_icon = "🔴"
        risk_color = "red"
        summary_reason = "Life-threatening symptoms detected requiring immediate medical attention."
    elif risk_score >= 6:
        risk_level = "HIGH"
        urgency = "Seek care within 24 hours"
        urgency_icon = "🟠"
        risk_color = "orange"
        summary_reason = "Serious symptoms detected that should be evaluated soon."
    elif risk_score >= 3:
        risk_level = "MODERATE"
        urgency = "Schedule appointment within 2-3 days"
        urgency_icon = "🟡"
        risk_color = "yellow"
        summary_reason = "Moderate symptoms detected that may need medical evaluation."
    else:
        risk_level = "LOW"
        urgency = "Monitor and follow-up if needed"
        urgency_icon = "🔵"
        risk_color = "blue"
        summary_reason = "Mild symptoms detected. Monitor and seek care if symptoms worsen."
    
    # Display summary box
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if risk_score == 0.0:
            st.success(f"**Risk Level:** {risk_level}")
        elif risk_score >= 9:
            st.error(f"**Risk Level:** {risk_level}")
        elif risk_score >= 6:
            st.warning(f"**Risk Level:** {risk_level}")
        elif risk_score >= 3:
            st.info(f"**Risk Level:** {risk_level}")
        else:
            st.info(f"**Risk Level:** {risk_level}")
    
    with col2:
        st.metric("Risk Score", f"{risk_score:.1f}/10")
    
    with col3:
        st.markdown(f"**{urgency_icon} Urgency:** {urgency}")
        st.caption(summary_reason)
    
    st.markdown("---")
    
    # ============================================
    # SECTION 2: WHAT CAUSED THIS RISK (Patient-Friendly)
    # ============================================
    st.markdown("## 🔍 What Caused This Risk?")
    
    # Extract top symptoms that drove the severity
    symptoms = normalized.get('symptoms', [])
    critical_issues = audit_report.get('critical_issues', [])
    high_issues = audit_report.get('high_issues', [])
    moderate_issues = audit_report.get('moderate_issues', [])
    
    # Get top contributing symptoms (patient-friendly)
    top_symptoms = _extract_top_symptoms_patient_friendly(symptoms, critical_issues, high_issues, moderate_issues)
    
    if top_symptoms:
        st.markdown("The following symptoms or findings contributed to this assessment:")
        for symptom in top_symptoms[:3]:  # Top 2-3
            st.markdown(f"• **{symptom}**")
    else:
        st.info("No specific symptoms were identified that require immediate attention.")
    
    # Extract vitals that may be concerning
    vitals = normalized.get('vitals', [])
    concerning_vitals = _extract_concerning_vitals_patient_friendly(vitals, risk_score)
    if concerning_vitals:
        st.markdown("\n**Vital Signs of Concern:**")
        for vital in concerning_vitals:
            st.markdown(f"• {vital}")
    
    st.markdown("---")
    
    # ============================================
    # SECTION 3: WHAT THIS MEANS
    # ============================================
    st.markdown("## 💡 What This Means")
    
    explanation = _generate_what_this_means(risk_score, critical_issues, high_issues, moderate_issues, symptoms)
    st.info(explanation)
    
    st.markdown("---")
    
    # ============================================
    # SECTION 4: WHAT YOU SHOULD DO NOW (Action Plan)
    # ============================================
    st.markdown("## ✅ What You Should Do Now")
    
    action_plan = _generate_action_plan(risk_score, urgency, critical_issues, high_issues, moderate_issues)
    for action in action_plan:
        st.markdown(action)
    
    st.markdown("---")
    
    # ============================================
    # SECTION 5: HOME CARE TIPS
    # ============================================
    if risk_score < 9:  # Show home care for non-critical cases
        st.markdown("## 🏠 Home Care Tips")
        
        home_care = _generate_home_care_tips(risk_score, symptoms)
        for tip in home_care:
            st.markdown(tip)
        
        st.markdown("---")
    
    # ============================================
    # SECTION 6: EMERGENCY WARNING SIGNS
    # ============================================
    if risk_score >= 6:  # Show for HIGH and CRITICAL
        st.markdown("## 🚨 Emergency Warning Signs")
        st.error("**Seek emergency care immediately if you experience any of the following:**")
        
        warning_signs = _generate_emergency_warning_signs(risk_score, symptoms)
        for sign in warning_signs:
            st.markdown(f"• {sign}")
        
        st.markdown("---")
    
    # ============================================
    # SECTION 7: CLINICIAN SUMMARY (Collapsible)
    # ============================================
    with st.expander("👨‍⚕️ **Clinician Summary** (For Healthcare Providers)", expanded=False):
        _display_clinician_summary(audit_report)
    
    st.markdown("---")
    
    # ============================================
    # DISCLAIMER
    # ============================================
    # ============================================
    # SECTION 8: DATA SUMMARY TABLES (Collapsible)
    # ============================================
    with st.expander("📊 **Detailed Data Summary** (Symptoms, Vitals, Lab Values)", expanded=False):
        ner_result = audit_report.get('ner_result', {})
        normalized = ner_result.get('normalized_entities', {}) if ner_result else {}
        
        # Symptom Summary Table
        symptoms = normalized.get('symptoms', [])
        if symptoms:
            st.subheader("📊 Symptom Summary")
            symptom_data = []
            for symptom in symptoms[:20]:  # Limit to 20
                symptom_data.append([symptom])
            
            if symptom_data:
                import pandas as pd
                df = pd.DataFrame(symptom_data, columns=['Symptom'])
                st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Vital Signs Summary
        vitals = normalized.get('vitals', [])
        if vitals:
            st.subheader("💓 Vital Signs Summary")
            vitals_data = []
            for vital in vitals:
                if isinstance(vital, dict):
                    vital_type = vital.get('type', 'Unknown')
                    if vital_type == 'blood_pressure':
                        vitals_data.append(['Blood Pressure', f"{vital.get('systolic', 'N/A')}/{vital.get('diastolic', 'N/A')} mmHg"])
                    else:
                        vitals_data.append([vital_type.replace('_', ' ').title(), str(vital.get('original', vital))])
                else:
                    vitals_data.append(['Vital Sign', str(vital)])
            
            if vitals_data:
                import pandas as pd
                df = pd.DataFrame(vitals_data, columns=['Type', 'Value'])
                st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Lab Value Summary
        lab_values = normalized.get('lab_values', [])
        if lab_values:
            st.subheader("🧪 Lab Value Summary")
            lab_data = []
            for lab in lab_values:
                if isinstance(lab, dict):
                    lab_data.append([
                        lab.get('test', 'Unknown'),
                        f"{lab.get('value', 'N/A')} {lab.get('unit', '')}"
                    ])
                else:
                    lab_data.append(['Lab Test', str(lab)])
            
            if lab_data:
                import pandas as pd
                df = pd.DataFrame(lab_data, columns=['Test', 'Value'])
                st.dataframe(df, use_container_width=True, hide_index=True)
    
    # ============================================
    # SECTION 9: STRUCTURED NEXT STEPS (If Available)
    # ============================================
    next_steps = audit_report.get('next_steps')
    if next_steps:
        st.markdown("---")
        st.markdown("## 📋 Recommendations & Next Steps (Actionable)")
        
        # Urgency badge
        urgency_level = next_steps.get('urgency_level', 'routine').upper()
        urgency_colors = {
            'IMMEDIATE': '🔴',
            'WITHIN_24_HOURS': '🟠',
            'WITHIN_72_HOURS': '🟡',
            'ROUTINE': '🟢'
        }
        urgency_icon = urgency_colors.get(urgency_level, '⚪')
        st.markdown(f"**{urgency_icon} Urgency Level:** {urgency_level.replace('_', ' ')}")
        
        # Action items table
        action_items = next_steps.get('action_items', [])
        if action_items:
            st.subheader("Action Items")
            for idx, action in enumerate(action_items, 1):
                priority = action.get('priority', 'medium').upper()
                action_type = action.get('action_type', 'Unknown')
                rationale = action.get('rationale', '')
                
                with st.expander(f"**{idx}. {action_type}** (Priority: {priority})", expanded=(priority in ['IMMEDIATE', 'HIGH'])):
                    st.markdown(f"**Rationale:** {rationale}")
                    
                    # Ordered items
                    ordered_items = action.get('ordered_items') or []
                    if ordered_items:
                        st.markdown("**Ordered Tests/Procedures:**")
                        for item in ordered_items:
                            st.markdown(f"• {item.get('item_name', 'Unknown')}")
                    
                    # Treatment recommendations
                    treatment = action.get('treatment_recommendations')
                    if treatment:
                        st.markdown("**Treatment Recommendations:**")
                        drug = treatment.get('drug', '')
                        dose = treatment.get('dose', '')
                        route = treatment.get('route', '')
                        if drug:
                            st.markdown(f"• **Drug:** {drug}")
                            if dose:
                                st.markdown(f"• **Dose:** {dose}")
                            if route:
                                st.markdown(f"• **Route:** {route}")
                        
                        contraindications = treatment.get('contraindications', [])
                        if contraindications:
                            st.warning(f"**Contraindications:** {', '.join(contraindications)}")
                        
                        if treatment.get('human_approval_required'):
                            st.error("**⚠️ Human approval required before starting this treatment**")
                    
                    # Monitoring parameters
                    monitoring = action.get('monitoring_parameters', [])
                    if monitoring:
                        st.markdown("**Monitoring:**")
                        for param in monitoring:
                            st.markdown(f"• {param.get('parameter', 'Unknown')}: {param.get('frequency', '')}")
                    
                    # Disposition
                    disposition = action.get('disposition', '')
                    if disposition:
                        st.markdown(f"**Disposition:** {disposition}")
                    
                    # Evidence IDs
                    evidence_ids = action.get('evidence_ids', [])
                    if evidence_ids:
                        st.caption(f"Evidence IDs: {', '.join([str(eid) for eid in evidence_ids[:3]])}")
        
        # Patient instructions
        patient_instructions = next_steps.get('patient_instructions', '')
        if patient_instructions:
            st.markdown("---")
            st.markdown("### 👤 Patient Instructions")
            st.info(patient_instructions)
        
        # Clinician note
        clinician_note = next_steps.get('clinician_note', '')
        if clinician_note:
            with st.expander("📝 Clinician Note", expanded=False):
                st.text(clinician_note)
    
    # ============================================
    # SECTION 10: DISCLAIMER
    # ============================================
    st.markdown("---")
    st.warning("""
    **Important Disclaimer:** 
    This is an automated medical assessment tool and does not replace professional medical judgment. 
    Always consult with a qualified healthcare provider for medical decisions. 
    In case of emergency, call your local emergency services immediately.
    """)


def _generate_next_steps(audit_report: Dict[str, Any]) -> List[str]:
    """Generate detailed next-step recommendations."""
    steps = []
    critical_issues = audit_report.get('critical_issues', [])
    high_issues = audit_report.get('high_issues', [])
    moderate_issues = audit_report.get('moderate_issues', [])
    
    if critical_issues:
        steps.append("**🚨 IMMEDIATE ACTIONS (Within 1 hour):**")
        for issue in critical_issues[:5]:
            agent = issue.get('agent', '')
            if agent == 'RedFlagChecker' or agent == 'PatternDetector':
                steps.append("• **Emergency Evaluation**: Patient requires immediate emergency department evaluation")
                steps.append("• **Vital Signs**: Monitor continuously (BP, HR, O2 sat, temperature)")
                steps.append("• **IV Access**: Establish IV access for potential medications")
            elif agent == 'DosageChecker':
                steps.append("• **Dosage Review**: Immediately review and correct medication dosages")
                steps.append("• **Pharmacy Consult**: Contact pharmacist for dosage verification")
            elif agent == 'InteractionChecker':
                steps.append("• **Drug Review**: Discontinue one of the interacting medications immediately")
                steps.append("• **Alternative Therapy**: Consider alternative medications")
    
    if high_issues:
        steps.append("\n**⚠️ URGENT EVALUATIONS (Within 24 hours):**")
        for issue in high_issues[:3]:
            agent = issue.get('agent', '')
            if agent == 'MissingTestsChecker':
                steps.append("• **Lab Orders**: Order missing essential laboratory tests before continuing medication")
                steps.append("• **Baseline Monitoring**: Establish baseline values for ongoing monitoring")
    
    if moderate_issues:
        steps.append("\n**ℹ️ FOLLOW-UP ACTIONS (Within 1 week):**")
        steps.append("• **Treatment Review**: Review treatment plan for guideline compliance")
        steps.append("• **Alternative Approaches**: Consider alternative therapeutic approaches if needed")
    
    if not steps:
        steps.append("• Continue routine monitoring as per standard protocol")
        steps.append("• Review treatment plan at next scheduled visit")
    
    return steps


def _extract_top_symptoms_patient_friendly(symptoms: List[str], critical_issues: List[Dict], 
                                           high_issues: List[Dict], moderate_issues: List[Dict]) -> List[str]:
    """Extract top 2-3 symptoms in patient-friendly language (no agent names, no technical terms)."""
    top_symptoms = []
    
    # Map technical terms to patient-friendly terms
    symptom_map = {
        'chest pain': 'Chest pain or discomfort',
        'shortness of breath': 'Difficulty breathing',
        'dyspnea': 'Difficulty breathing',
        'sob': 'Difficulty breathing',
        'fever': 'Fever',
        'high fever': 'High fever',
        'temperature': 'Fever',
        'cough': 'Cough',
        'sputum': 'Cough with phlegm',
        'yellow sputum': 'Cough with yellow or green phlegm',
        'headache': 'Headache',
        'severe headache': 'Severe headache',
        'neck stiffness': 'Stiff neck',
        'confusion': 'Confusion or difficulty thinking clearly',
        'nausea': 'Nausea',
        'vomiting': 'Vomiting',
        'diarrhea': 'Diarrhea',
        'dizziness': 'Dizziness',
        'sweating': 'Excessive sweating',
        'glucose': 'High or low blood sugar',
        'blood pressure': 'High or low blood pressure',
        'oxygen': 'Low oxygen levels',
        'heart rate': 'Rapid or irregular heartbeat'
    }
    
    # Extract from symptoms list
    for symptom in symptoms[:5]:
        symptom_lower = str(symptom).lower()
        # Find patient-friendly mapping
        for tech_term, friendly_term in symptom_map.items():
            if tech_term in symptom_lower:
                if friendly_term not in top_symptoms:
                    top_symptoms.append(friendly_term)
                break
        else:
            # If no mapping found, capitalize and use as-is
            friendly = str(symptom).replace('_', ' ').title()
            if friendly not in top_symptoms:
                top_symptoms.append(friendly)
    
    # Extract from issues (but remove agent references)
    all_issues = critical_issues + high_issues + moderate_issues
    for issue in all_issues[:3]:
        message = issue.get('message', '')
        # Remove agent names and technical terms
        message_clean = message
        for agent_name in ['RedFlagChecker', 'DosageChecker', 'InteractionChecker', 'MissingTestsChecker', 'GuidelineComplianceChecker']:
            message_clean = message_clean.replace(agent_name, '')
        
        # Extract key symptom from message
        for tech_term, friendly_term in symptom_map.items():
            if tech_term in message_clean.lower() and friendly_term not in top_symptoms:
                top_symptoms.append(friendly_term)
                break
    
    return top_symptoms[:3]  # Return top 2-3


def _extract_concerning_vitals_patient_friendly(vitals: List[Any], risk_score: float) -> List[str]:
    """Extract concerning vital signs in patient-friendly language."""
    concerning = []
    
    for vital in vitals:
        if isinstance(vital, dict):
            vital_type = str(vital.get('type', '')).lower()
            
            if 'blood_pressure' in vital_type or 'bp' in vital_type:
                systolic = vital.get('systolic') or (vital.get('value', [])[0] if isinstance(vital.get('value'), list) and len(vital.get('value', [])) >= 1 else None)
                diastolic = vital.get('diastolic') or (vital.get('value', [])[1] if isinstance(vital.get('value'), list) and len(vital.get('value', [])) >= 2 else None)
                
                if systolic and diastolic:
                    if systolic > 180 or diastolic > 120:
                        concerning.append(f"Very high blood pressure ({systolic}/{diastolic})")
                    elif systolic < 90:
                        concerning.append(f"Low blood pressure ({systolic}/{diastolic})")
            
            elif 'heart' in vital_type or 'hr' in vital_type or 'pulse' in vital_type:
                hr = vital.get('value') or vital.get('heart_rate') or vital.get('pulse')
                if hr:
                    hr_val = float(hr) if isinstance(hr, (int, float, str)) else None
                    if hr_val:
                        if hr_val > 120:
                            concerning.append(f"Rapid heart rate ({int(hr_val)} beats per minute)")
                        elif hr_val < 50:
                            concerning.append(f"Slow heart rate ({int(hr_val)} beats per minute)")
            
            elif 'oxygen' in vital_type or 'spo2' in vital_type:
                o2 = vital.get('value') or vital.get('oxygen') or vital.get('spo2')
                if o2:
                    o2_val = float(o2) if isinstance(o2, (int, float, str)) else None
                    if o2_val and o2_val < 95:
                        concerning.append(f"Low oxygen level ({o2_val}%)")
            
            elif 'temperature' in vital_type or 'temp' in vital_type or 'fever' in vital_type:
                temp = vital.get('value') or vital.get('temperature') or vital.get('temp')
                if temp:
                    temp_val = float(temp) if isinstance(temp, (int, float, str)) else None
                    if temp_val:
                        if temp_val >= 101.5:
                            concerning.append(f"High fever ({temp_val}°F)")
                        elif temp_val >= 99.5:
                            concerning.append(f"Elevated temperature ({temp_val}°F)")
    
    return concerning


def _generate_what_this_means(risk_score: float, critical_issues: List[Dict], high_issues: List[Dict],
                              moderate_issues: List[Dict], symptoms: List[str]) -> str:
    """Generate patient-friendly explanation of what the assessment means."""
    
    if risk_score == 0.0:
        return """
        **Good News:** Your assessment shows no concerning findings. Your symptoms, vital signs, and lab values 
        are all within normal ranges. Continue with your regular health maintenance and routine check-ups.
        """
    
    elif risk_score >= 9:
        return """
        **This is a medical emergency.** The symptoms and findings detected indicate a potentially life-threatening 
        condition that requires immediate medical attention. Possible conditions include heart attack, stroke, severe 
        infection, or other critical emergencies. **Do not delay—seek emergency care right away.**
        """
    
    elif risk_score >= 6:
        return """
        **This requires urgent medical evaluation.** The symptoms detected suggest a serious but not immediately 
        life-threatening condition. Possible causes include pneumonia, severe infection, significant medication issues, 
        or other serious health problems. You should be evaluated by a healthcare provider within 24 hours.
        """
    
    elif risk_score >= 3:
        return """
        **This may need medical attention.** The symptoms detected suggest a moderate health concern that could 
        benefit from medical evaluation. Possible causes include infection, medication side effects, or other 
        treatable conditions. Schedule an appointment with your healthcare provider within 2-3 days.
        """
    
    else:  # LOW (1-2)
        return """
        **This is likely a mild, self-limiting condition.** The symptoms detected are typically associated with 
        minor illnesses like the common cold, seasonal allergies, or mild viral infections. These usually resolve 
        on their own with rest and home care. Monitor your symptoms and seek care if they worsen.
        """


def _generate_action_plan(risk_score: float, urgency: str, critical_issues: List[Dict], 
                         high_issues: List[Dict], moderate_issues: List[Dict]) -> List[str]:
    """Generate clear action plan with timeline."""
    actions = []
    
    if risk_score >= 9:
        actions.append("**🚨 IMMEDIATE (Right Now):**")
        actions.append("• Call 911 or go to the nearest emergency room immediately")
        actions.append("• Do not drive yourself—have someone take you or call an ambulance")
        actions.append("• Bring a list of your current medications")
        actions.append("• Inform emergency staff about your symptoms")
    
    elif risk_score >= 6:
        actions.append("**🟠 WITHIN 24 HOURS:**")
        actions.append("• Contact your healthcare provider's office today to schedule an urgent appointment")
        actions.append("• If your provider is unavailable, consider urgent care or emergency department")
        actions.append("• Monitor your symptoms closely—seek immediate care if they worsen")
        actions.append("• Bring a list of your medications to your appointment")
    
    elif risk_score >= 3:
        actions.append("**🟡 WITHIN 2-3 DAYS:**")
        actions.append("• Schedule an appointment with your healthcare provider")
        actions.append("• Monitor your symptoms and note any changes")
        actions.append("• Continue any current medications unless advised otherwise")
        actions.append("• Seek care sooner if symptoms worsen or new symptoms develop")
    
    else:
        actions.append("**🔵 MONITOR AND FOLLOW-UP:**")
        actions.append("• Continue monitoring your symptoms at home")
        actions.append("• Use home care measures (see Home Care Tips below)")
        actions.append("• Contact your healthcare provider if symptoms persist for more than 7-10 days")
        actions.append("• Seek care immediately if symptoms worsen or you develop new concerning symptoms")
    
    return actions


def _generate_home_care_tips(risk_score: float, symptoms: List[str]) -> List[str]:
    """Generate home care advice based on severity and symptoms."""
    tips = []
    
    # Hydration
    tips.append("**💧 Stay Hydrated:**")
    tips.append("• Drink plenty of water (8-10 glasses per day)")
    tips.append("• Avoid caffeinated and alcoholic beverages")
    tips.append("• If you have diarrhea or vomiting, consider oral rehydration solutions")
    
    # Safe medications
    tips.append("\n**💊 Safe Medications (Use as Directed):**")
    tips.append("• **For fever or pain:** Acetaminophen (Tylenol) 500-1000 mg every 4-6 hours as needed")
    tips.append("  - Maximum: 3000 mg per day")
    tips.append("  - Do not exceed this limit to avoid liver damage")
    tips.append("• **For cough:** Over-the-counter cough suppressants or expectorants as directed")
    tips.append("• **For nasal congestion:** Saline nasal sprays or decongestants (use for no more than 3 days)")
    tips.append("• **Important:** Check with your healthcare provider before taking new medications if you have other medical conditions")
    
    # Rest
    tips.append("\n**😴 Rest:**")
    tips.append("• Get plenty of rest and sleep")
    tips.append("• Avoid strenuous activities until symptoms improve")
    tips.append("• Take time off work or school if needed")
    
    # Symptom monitoring
    tips.append("\n**📊 Monitor Your Symptoms:**")
    tips.append("• Check your temperature twice daily")
    tips.append("• Note any changes in your symptoms")
    tips.append("• Keep a simple log: date, time, symptom, severity (mild/moderate/severe)")
    tips.append("• Watch for warning signs (see Emergency Warning Signs section if applicable)")
    
    # Additional tips based on symptoms
    symptom_lower = [str(s).lower() for s in symptoms]
    if any('cough' in s for s in symptom_lower):
        tips.append("\n**For Cough:**")
        tips.append("• Use a humidifier or take steamy showers")
        tips.append("• Avoid irritants like smoke and strong odors")
        tips.append("• Stay hydrated to help thin mucus")
    
    if any('fever' in s or 'temperature' in s for s in symptom_lower):
        tips.append("\n**For Fever:**")
        tips.append("• Dress in light, breathable clothing")
        tips.append("• Use cool compresses on forehead and neck")
        tips.append("• Take acetaminophen or ibuprofen as directed (check with provider first)")
    
    return tips


def _generate_emergency_warning_signs(risk_score: float, symptoms: List[str]) -> List[str]:
    """Generate 6-10 emergency warning signs that require ER visit."""
    warning_signs = [
        "**Severe difficulty breathing** or inability to catch your breath",
        "**Chest pain or pressure** that is severe, crushing, or radiates to arm/jaw",
        "**Confusion, disorientation, or difficulty speaking**",
        "**Severe headache** that comes on suddenly (thunderclap headache)",
        "**Fainting or loss of consciousness**",
        "**Severe abdominal pain** that is constant and worsening",
        "**Signs of severe dehydration:** no urination for 8+ hours, extreme thirst, dry mouth, dizziness when standing",
        "**High fever** (above 103°F) that doesn't respond to medication",
        "**Severe allergic reaction:** difficulty breathing, swelling of face/throat, hives",
        "**One-sided weakness, facial droop, or slurred speech** (possible stroke)"
    ]
    
    # Add specific warnings based on detected symptoms
    symptom_lower = [str(s).lower() for s in symptoms]
    
    if any('chest' in s for s in symptom_lower):
        warning_signs.insert(0, "**Chest pain that worsens** or spreads to your arm, neck, or jaw")
    
    if any('breath' in s or 'sob' in s or 'dyspnea' in s for s in symptom_lower):
        warning_signs.insert(0, "**Breathing becomes more difficult** or you feel like you can't get enough air")
    
    if any('neck' in s and 'stiff' in s for s in symptom_lower):
        warning_signs.insert(0, "**Neck stiffness with fever and severe headache** (possible meningitis)")
    
    return warning_signs[:10]  # Return up to 10


def _display_clinician_summary(audit_report: Dict[str, Any]):
    """Display detailed clinician-only summary with technical details."""
    st.markdown("### Technical Assessment Details")
    
    # Risk score and severity
    risk_score = audit_report.get('risk_score', 0.0)
    st.markdown(f"**Overall Risk Score:** {risk_score:.2f}/10")
    
    # Agent outputs (technical)
    agent_outputs = audit_report.get('agent_outputs', [])
    if agent_outputs:
        st.markdown("#### Agent Analysis Results")
        for agent in agent_outputs:
            agent_name = agent.get('agent', 'Unknown')
            status = agent.get('status', 'ok')
            score = agent.get('score', 0.0)
            message = agent.get('message', '')
            
            st.markdown(f"**{agent_name}:** {status.upper()} (Score: {score:.2f}/10)")
            st.caption(f"{message}")
            
            # Show details if available
            details = agent.get('details', {})
            if details:
                with st.expander(f"View {agent_name} Details"):
                    st.json(details)
    
    # Critical/High/Moderate issues with agent references
    critical_issues = audit_report.get('critical_issues', [])
    high_issues = audit_report.get('high_issues', [])
    moderate_issues = audit_report.get('moderate_issues', [])
    
    if critical_issues:
        st.markdown("#### Critical Issues")
        for issue in critical_issues:
            st.markdown(f"**{issue.get('agent', 'Unknown')}:** {issue.get('message', '')}")
            if issue.get('why_risky'):
                st.caption(f"Reasoning: {issue.get('why_risky', '')}")
    
    if high_issues:
        st.markdown("#### High Priority Issues")
        for issue in high_issues:
            st.markdown(f"**{issue.get('agent', 'Unknown')}:** {issue.get('message', '')}")
    
    if moderate_issues:
        st.markdown("#### Moderate Issues")
        for issue in moderate_issues[:5]:
            st.markdown(f"**{issue.get('agent', 'Unknown')}:** {issue.get('message', '')}")
    
    # Suggested labs/imaging from next_steps
    next_steps = audit_report.get('next_steps')
    if next_steps:
        items = next_steps.get('items', [])
        suggested_tests = []
        for item in items:
            ordered_items = item.get('ordered_items') or []
            if ordered_items:
                for ordered in ordered_items:
                    test_name = ordered.get('name', '')
                    test_type = ordered.get('type', '')
                    if test_name and test_name not in suggested_tests:
                        suggested_tests.append(f"{test_name} ({test_type})")
        
        if suggested_tests:
            st.markdown("#### Suggested Labs/Imaging")
            for test in suggested_tests:
                st.markdown(f"• {test}")
    
    # Suspected categories
    st.markdown("#### Suspected Clinical Categories")
    categories = _extract_suspected_categories(audit_report)
    for category in categories:
        st.markdown(f"• {category}")
    
    # NER Results Summary
    ner_result = audit_report.get('ner_result', {})
    normalized = ner_result.get('normalized_entities', {}) if ner_result else {}
    
    st.markdown("#### Extracted Clinical Data")
    col1, col2 = st.columns(2)
    with col1:
        symptoms = normalized.get('symptoms', [])
        st.markdown(f"**Symptoms:** {len(symptoms)} detected")
        if symptoms:
            st.caption(", ".join([str(s) for s in symptoms[:10]]))
        
        vitals = normalized.get('vitals', [])
        st.markdown(f"**Vital Signs:** {len(vitals)} detected")
    
    with col2:
        lab_values = normalized.get('lab_values', [])
        st.markdown(f"**Lab Values:** {len(lab_values)} detected")
        
        drugs = normalized.get('drugs', [])
        st.markdown(f"**Medications:** {len(drugs)} detected")


def _extract_suspected_categories(audit_report: Dict[str, Any]) -> List[str]:
    """Extract suspected clinical categories from issues."""
    categories = []
    
    critical_issues = audit_report.get('critical_issues', [])
    high_issues = audit_report.get('high_issues', [])
    moderate_issues = audit_report.get('moderate_issues', [])
    
    all_issues = critical_issues + high_issues + moderate_issues
    
    # Check for category keywords
    category_keywords = {
        'Cardiac': ['chest pain', 'heart', 'cardiac', 'myocardial', 'angina'],
        'Respiratory': ['breath', 'cough', 'sputum', 'oxygen', 'dyspnea', 'sob'],
        'Neurological': ['headache', 'confusion', 'stroke', 'seizure', 'weakness', 'facial droop'],
        'Infectious': ['fever', 'infection', 'sepsis', 'meningitis'],
        'Endocrine/Metabolic': ['glucose', 'diabetes', 'diabetic', 'hyperglycemia', 'hypoglycemia'],
        'Gastrointestinal': ['nausea', 'vomiting', 'diarrhea', 'abdominal pain'],
        'Medication-Related': ['dosage', 'interaction', 'drug', 'medication']
    }
    
    issue_text = ' '.join([issue.get('message', '') + ' ' + str(issue.get('details', '')) for issue in all_issues]).lower()
    
    for category, keywords in category_keywords.items():
        if any(keyword in issue_text for keyword in keywords):
            categories.append(category)
    
    if not categories:
        categories.append("General Medical Evaluation")
    
    return categories


def _generate_patient_explanation(audit_report: Dict[str, Any]) -> str:
    """Generate patient-friendly explanation of the audit report."""
    risk_score = audit_report.get('risk_score', 0.0)
    critical_issues = audit_report.get('critical_issues', [])
    high_issues = audit_report.get('high_issues', [])
    
    if risk_score >= 9:
        explanation = "🚨 **URGENT ATTENTION REQUIRED**: This report indicates critical medical concerns that require immediate medical attention. Please contact your healthcare provider or seek emergency care right away."
    elif risk_score >= 6:
        explanation = "⚠️ **HIGH PRIORITY**: This report shows important medical concerns that should be addressed soon. Please schedule an appointment with your healthcare provider within 24-48 hours."
    elif risk_score >= 3:
        explanation = "ℹ️ **MODERATE CONCERNS**: This report identifies some areas that may need attention. Please discuss these findings with your healthcare provider at your next visit."
    else:
        explanation = "✅ **LOW RISK**: This report shows that most aspects of your care appear safe. Continue following your current treatment plan and regular monitoring."
    
    if critical_issues:
        explanation += f"\n\n**Critical Issues Found**: {len(critical_issues)} critical issue(s) detected. These require immediate medical evaluation."
    elif high_issues:
        explanation += f"\n\n**High Priority Issues**: {len(high_issues)} high-priority issue(s) that should be addressed soon."
    
    explanation += "\n\n*This is an automated analysis. Always consult with your healthcare provider for medical decisions.*"
    
    return explanation


def generate_pdf_report(audit_report: Dict[str, Any]) -> bytes:
    """Generate comprehensive PDF report from audit results."""
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from datetime import datetime
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12
    )
    
    # Title
    story.append(Paragraph("MedInsight - Medical Audit Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Risk Score with color coding
    risk_score = audit_report.get('risk_score', 0.0)
    if risk_score == 0.0:
        risk_color = colors.green
        risk_level = "SAFE"
    elif risk_score >= 9:
        risk_color = colors.red
        risk_level = "CRITICAL"
    elif risk_score >= 6:
        risk_color = colors.orange
        risk_level = "HIGH"
    elif risk_score >= 3:
        risk_color = colors.yellow
        risk_level = "MODERATE"
    else:
        risk_color = colors.green
        risk_level = "LOW"
    
    story.append(Paragraph(f"<b>Overall Risk Score: <font color='{risk_color.hexval()}'>{risk_score:.2f}/10 ({risk_level})</font></b>", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    # Extract NER data
    ner_result = audit_report.get('ner_result', {})
    normalized = ner_result.get('normalized_entities', {}) if ner_result else {}
    
    # Symptom Summary Table
    symptoms = normalized.get('symptoms', [])
    if symptoms:
        story.append(Paragraph("<b>📊 Symptom Summary</b>", heading_style))
        symptom_data = [['Symptom']]
        for symptom in symptoms[:15]:
            symptom_data.append([str(symptom)])
        
        if len(symptom_data) > 1:
            table = Table(symptom_data, colWidths=[5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 0.2*inch))
    
    # Vital Signs Summary
    vitals = normalized.get('vitals', [])
    if vitals:
        story.append(Paragraph("<b>💓 Vital Signs Summary</b>", heading_style))
        vitals_data = [['Type', 'Value']]
        for vital in vitals[:10]:
            if isinstance(vital, dict):
                vital_type = vital.get('type', 'Unknown')
                if vital_type == 'blood_pressure':
                    vitals_data.append(['Blood Pressure', f"{vital.get('systolic', 'N/A')}/{vital.get('diastolic', 'N/A')} mmHg"])
                else:
                    vitals_data.append([vital_type.replace('_', ' ').title(), str(vital.get('original', vital))[:50]])
            else:
                vitals_data.append(['Vital Sign', str(vital)[:50]])
        
        if len(vitals_data) > 1:
            table = Table(vitals_data, colWidths=[2.5*inch, 2.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 0.2*inch))
    
    # Lab Value Summary
    lab_values = normalized.get('lab_values', [])
    if lab_values:
        story.append(Paragraph("<b>🧪 Lab Value Summary</b>", heading_style))
        lab_data = [['Test', 'Value']]
        for lab in lab_values[:10]:
            if isinstance(lab, dict):
                lab_data.append([
                    lab.get('test', 'Unknown'),
                    f"{lab.get('value', 'N/A')} {lab.get('unit', '')}"
                ])
            else:
                lab_data.append(['Lab Test', str(lab)[:50]])
        
        if len(lab_data) > 1:
            table = Table(lab_data, colWidths=[2.5*inch, 2.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 0.2*inch))
    
    # Patient-Friendly Explanation
    patient_explanation = _generate_patient_explanation(audit_report)
    story.append(Paragraph("<b>📖 Patient-Friendly Explanation</b>", heading_style))
    story.append(Paragraph(patient_explanation, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Critical Issues with "Why this is risky"
    critical_issues = audit_report.get('critical_issues', [])
    if critical_issues:
        story.append(Paragraph(f"<b><font color='red'>🚨 Critical Issues ({len(critical_issues)})</font></b>", heading_style))
        for issue in critical_issues[:10]:  # Limit to 10
            agent = issue.get('agent', 'Unknown')
            message = issue.get('message', '')
            why_risky = issue.get('why_risky', '')
            
            story.append(Paragraph(f"<b>{agent}:</b> {message}", styles['Normal']))
            if why_risky:
                story.append(Paragraph(f"<i>⚠️ Why this is risky:</i> {why_risky}", styles['Normal']))
            
            # Confidence Score
            confidence = issue.get('confidence')
            confidence_explanation = issue.get('confidence_explanation', '')
            if confidence is not None:
                story.append(Paragraph(f"<b>Confidence:</b> {confidence*100:.0f}%", styles['Normal']))
                if confidence_explanation:
                    story.append(Paragraph(f"<i>{confidence_explanation[:150]}...</i>", styles['Normal']))
            
            # RAG Evidence
            evidence = issue.get('evidence', [])
            if evidence:
                story.append(Paragraph("<b>Retrieved Guidelines:</b>", styles['Normal']))
                for ev in evidence[:2]:
                    ev_text = str(ev)[:200] + "..." if len(str(ev)) > 200 else str(ev)
                    story.append(Paragraph(f"• {ev_text}", styles['Normal']))
            
            story.append(Spacer(1, 0.1*inch))
        story.append(Spacer(1, 0.2*inch))
    
    # High Issues
    high_issues = audit_report.get('high_issues', [])
    if high_issues:
        story.append(Paragraph(f"<b><font color='orange'>High Priority Issues ({len(high_issues)})</font></b>", heading_style))
        for issue in high_issues[:10]:
            agent = issue.get('agent', 'Unknown')
            message = issue.get('message', '')
            story.append(Paragraph(f"<b>{agent}:</b> {message}", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        story.append(Spacer(1, 0.2*inch))
    
    # Moderate Issues
    moderate_issues = audit_report.get('moderate_issues', [])
    if moderate_issues:
        story.append(Paragraph(f"<b><font color='yellow'>Moderate Issues ({len(moderate_issues)})</font></b>", heading_style))
        for issue in moderate_issues[:5]:
            agent = issue.get('agent', 'Unknown')
            message = issue.get('message', '')
            story.append(Paragraph(f"<b>{agent}:</b> {message}", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        story.append(Spacer(1, 0.2*inch))
    
    # Safe Items
    safe_items = audit_report.get('safe_items', [])
    if safe_items:
        story.append(Paragraph(f"<b><font color='green'>Safe Items ({len(safe_items)})</font></b>", heading_style))
        for item in safe_items[:5]:
            agent = item.get('agent', 'Unknown')
            message = item.get('message', '')
            story.append(Paragraph(f"<b>{agent}:</b> {message}", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        story.append(Spacer(1, 0.2*inch))
    
    # Recommendations
    recommendations = audit_report.get('recommendations', [])
    if recommendations:
        story.append(Paragraph("<b>💡 Recommendations</b>", heading_style))
        for i, rec in enumerate(recommendations[:10], 1):
            story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        story.append(Spacer(1, 0.2*inch))
    
    # Next Steps (Actionable)
    next_steps = audit_report.get('next_steps')
    if next_steps:
        story.append(Paragraph("<b>📋 Recommendations & Next Steps (Actionable)</b>", heading_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Urgency badge
        urgency = next_steps.get('urgency_level', 'routine')
        urgency_labels = {
            'immediate': '🔴 IMMEDIATE',
            '24h': '🟠 WITHIN 24 HOURS',
            '72h': '🟡 WITHIN 72 HOURS',
            'routine': '🟢 ROUTINE'
        }
        story.append(Paragraph(f"<b>{urgency_labels.get(urgency, urgency.upper())}</b>", heading_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Summary
        story.append(Paragraph(f"<b>Summary:</b> {next_steps.get('summary', '')}", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        
        # Disclaimer
        story.append(Paragraph(f"<i><font color='red'>{next_steps.get('disclaimer', '')}</font></i>", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Action items
        items = next_steps.get('items', [])
        for idx, item in enumerate(items[:10], 1):
            story.append(Paragraph(f"<b>{idx}. {item.get('title', '')}</b> ({item.get('priority', '').upper()})", heading_style))
            story.append(Paragraph(f"<b>Action Type:</b> {item.get('action_type', '')}", styles['Normal']))
            story.append(Paragraph(f"<b>Recommended by:</b> {item.get('recommended_by_agent', '')}", styles['Normal']))
            story.append(Paragraph(f"<b>Rationale:</b> {item.get('rationale', '')}", styles['Normal']))
            
            # Ordered items
            ordered_items = item.get('ordered_items') or []
            if ordered_items:
                story.append(Paragraph("<b>Ordered Items:</b>", styles['Normal']))
                for ordered in ordered_items:
                    story.append(Paragraph(f"  • {ordered.get('name', '')} ({ordered.get('type', '')}) - {ordered.get('urgency', '')}", styles['Normal']))
            
            # Treatment recommendations
            if item.get('treatment_recommendations'):
                story.append(Paragraph("<b>Treatment Recommendations:</b>", styles['Normal']))
                for treatment in item.get('treatment_recommendations', []):
                    approval_text = " ⚠️ HUMAN APPROVAL REQUIRED" if treatment.get('human_approval_required') else ""
                    story.append(Paragraph(f"  • {treatment.get('drug', '')}: {treatment.get('dose', '')}{approval_text}", styles['Normal']))
                    if treatment.get('contraindications'):
                        story.append(Paragraph(f"    Contraindications: {', '.join(treatment.get('contraindications', []))}", styles['Normal']))
            
            # Monitoring parameters
            if item.get('monitoring_parameters'):
                story.append(Paragraph("<b>Monitoring Parameters:</b>", styles['Normal']))
                for monitor in item.get('monitoring_parameters', []):
                    story.append(Paragraph(f"  • {monitor.get('parameter', '')}: Target {monitor.get('target', '')}, Frequency {monitor.get('frequency', '')}", styles['Normal']))
            
            story.append(Paragraph(f"<b>Disposition:</b> {item.get('disposition', '')}", styles['Normal']))
            story.append(Paragraph(f"<b>Clinical Confidence:</b> {item.get('clinical_confidence', '').upper()}", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Patient instructions
        story.append(Paragraph("<b>👤 Patient Instructions</b>", heading_style))
        story.append(Paragraph(next_steps.get('patient_instructions', ''), styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Clinician note
        story.append(Paragraph("<b>📝 Clinician Note</b>", heading_style))
        story.append(Paragraph(next_steps.get('clinician_note', ''), styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
    
    # What the doctor should check next (fallback if no next_steps)
    if not next_steps:
        story.append(Paragraph("<b>🔍 What the Doctor Should Check Next</b>", heading_style))(Paragraph("<b>🔍 What the Doctor Should Check Next</b>", heading_style))
    
    critical_issues = audit_report.get('critical_issues', [])
    high_issues = audit_report.get('high_issues', [])
    
    if critical_issues:
        story.append(Paragraph("<b>IMMEDIATE ACTIONS:</b>", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        for issue in critical_issues[:3]:
            agent = issue.get('agent', '')
            if agent == 'RedFlagChecker':
                story.append(Paragraph("• Evaluate patient for emergency conditions immediately", styles['Normal']))
            elif agent == 'DosageChecker':
                story.append(Paragraph("• Review and correct medication dosages urgently", styles['Normal']))
            elif agent == 'InteractionChecker':
                story.append(Paragraph("• Assess drug interactions and consider alternative medications", styles['Normal']))
            story.append(Spacer(1, 0.05*inch))
    
    if high_issues:
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("<b>URGENT EVALUATIONS:</b>", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        for issue in high_issues[:2]:
            agent = issue.get('agent', '')
            if agent == 'MissingTestsChecker':
                story.append(Paragraph("• Order missing essential laboratory tests", styles['Normal']))
                story.append(Spacer(1, 0.05*inch))
    
    if not critical_issues and not high_issues:
        story.append(Paragraph("• Continue routine monitoring", styles['Normal']))
        story.append(Paragraph("• Review treatment plan at next visit", styles['Normal']))
    
    story.append(Spacer(1, 0.2*inch))
    
    # Next-Step Recommendations
    next_steps = _generate_next_steps(audit_report)
    if next_steps:
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("<b>📋 Detailed Next Steps</b>", heading_style))
        for step in next_steps:
            story.append(Paragraph(step.replace('**', '').replace('🚨', '').replace('⚠️', '').replace('ℹ️', ''), styles['Normal']))
            story.append(Spacer(1, 0.05*inch))
    
    # Agent Outputs Summary with Confidence
    agent_outputs = audit_report.get('agent_outputs', [])
    if agent_outputs:
        story.append(PageBreak())
        story.append(Paragraph("<b>Agent Analysis Summary</b>", heading_style))
        
        # Create table with confidence
        data = [['Agent', 'Status', 'Score', 'Confidence']]
        for agent in agent_outputs:
            confidence = agent.get('confidence', 0.0) if isinstance(agent, dict) else 0.0
            if isinstance(agent, dict) and 'details' in agent:
                llm_explanation = agent.get('details', {}).get('llm_explanation', {})
                if isinstance(llm_explanation, dict):
                    confidence_str = llm_explanation.get('confidence', 'medium')
                else:
                    confidence_str = f"{confidence*100:.0f}%" if confidence else "N/A"
            else:
                confidence_str = f"{confidence*100:.0f}%" if confidence else "N/A"
            
            data.append([
                agent.get('agent', 'Unknown') if isinstance(agent, dict) else str(agent),
                agent.get('status', 'ok').upper() if isinstance(agent, dict) else 'OK',
                f"{agent.get('score', 0.0):.2f}" if isinstance(agent, dict) else "0.00",
                confidence_str
            ])
        
        table = Table(data, colWidths=[2.5*inch, 1.2*inch, 1*inch, 1.3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 0.2*inch))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

