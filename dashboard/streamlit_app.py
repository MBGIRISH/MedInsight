"""
MedInsight Streamlit Dashboard
Main application for medical audit visualization and interaction.
"""
import streamlit as st
import requests
import json
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add paths for imports
dashboard_dir = Path(__file__).parent
sys.path.insert(0, str(dashboard_dir))
sys.path.insert(0, str(dashboard_dir.parent))

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="MedInsight - AI Clinical Notes Auditor",
    page_icon="🏥",
    layout="wide"
)

# Import components with error handling
try:
    from components.uploader import file_uploader
    from components.report_viewer import (
        display_ner_results,
        display_agent_outputs,
        display_final_report,
        generate_pdf_report
    )
    from components.charts import (
        risk_score_gauge,
        agent_status_chart,
        issue_severity_pie,
        medication_timeline,
        lab_trend_chart
    )
    COMPONENTS_LOADED = True
except ImportError as e:
    COMPONENTS_LOADED = False
    st.error(f"❌ Error importing components: {e}")
    st.info("Please ensure all component files exist in the components/ directory")
    st.info("The app will continue with limited functionality.")

# API endpoint
API_BASE_URL = st.sidebar.text_input(
    "API Base URL",
    value="http://localhost:8000",
    help="Enter the FastAPI backend URL"
)

# Initialize session state
if 'audit_result' not in st.session_state:
    st.session_state.audit_result = None
if 'uploaded_text' not in st.session_state:
    st.session_state.uploaded_text = None


def call_api_ingest(file_bytes: bytes, file_name: str, file_type: str) -> Optional[Dict]:
    """Call ingestion API."""
    try:
        files = {'file': (file_name, file_bytes, file_type)}
        response = requests.post(f"{API_BASE_URL}/api/ingest", files=files, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error calling ingestion API: {str(e)}")
        return None


def call_api_audit(file_bytes: bytes, file_name: str, file_type: str) -> Optional[Dict]:
    """Call audit API with file upload."""
    try:
        files = {'file': (file_name, file_bytes, file_type)}
        response = requests.post(f"{API_BASE_URL}/api/audit", files=files, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error calling audit API: {str(e)}")
        return None


def call_api_audit_text(text: str) -> Optional[Dict]:
    """Call audit API with text input."""
    try:
        payload = {"text": text}
        response = requests.post(
            f"{API_BASE_URL}/api/audit/text",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error calling audit API: {str(e)}")
        return None


def main():
    """Main dashboard application."""
    # Show a simple message first to verify Streamlit is working
    st.title("🏥 MedInsight - AI Clinical Notes Auditor")
    st.markdown("---")
    
    # Debug info (can be removed later)
    if st.sidebar.checkbox("Show Debug Info", value=False):
        st.sidebar.write(f"Components Loaded: {COMPONENTS_LOADED}")
        st.sidebar.write(f"Python: {sys.version}")
        st.sidebar.write(f"Streamlit: {st.__version__}")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Upload & Audit", "View Results", "Analytics"]
    )
    
    # Health check
    try:
        health_response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        if health_response.status_code == 200:
            st.sidebar.success("✅ API Connected")
        else:
            st.sidebar.warning("⚠️ API Status Unknown")
    except Exception as e:
        st.sidebar.error(f"❌ API Not Available")
        st.sidebar.caption(f"Error: {str(e)[:50]}")
    
    # Main content
    if page == "Upload & Audit":
        st.header("📤 Upload Medical Document or Enter Text")
        
        # Tabs for file upload vs text input
        input_method = st.radio(
            "Choose input method:",
            ["📄 Upload File", "✍️ Enter Text"],
            horizontal=True
        )
        
        if input_method == "📄 Upload File":
            # File upload
            if COMPONENTS_LOADED:
                upload_result = file_uploader()
            else:
                st.error("File upload component not available. Please use text input.")
                upload_result = None
            
            if upload_result:
                file_bytes, file_name, file_type = upload_result
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔍 Extract Text & Entities", type="primary"):
                        with st.spinner("Extracting text and entities..."):
                            result = call_api_ingest(file_bytes, file_name, file_type)
                            if result:
                                st.session_state.uploaded_text = result.get('text', '')
                                st.success("✅ Extraction complete!")
                                
                                # Display extracted text
                                st.subheader("Extracted Text")
                                st.text_area("", value=st.session_state.uploaded_text, height=200, key="extracted_text")
                                
                                # Display NER results
                                if result.get('ner_result'):
                                    display_ner_results(result.get('ner_result'))
                
                with col2:
                    if st.button("🚀 Run Complete Audit", type="primary"):
                        with st.spinner("Running complete audit pipeline..."):
                            result = call_api_audit(file_bytes, file_name, file_type)
                            if result:
                                st.session_state.audit_result = result
                                st.success("✅ Audit complete!")
                                st.balloons()
        
        else:  # Text input
            st.subheader("Enter Clinical Text")
            text_input = st.text_area(
                "Paste or type clinical notes, prescription, or medical text:",
                height=200,
                placeholder="Example: Patient: John Doe, Age: 68. Prescription: Warfarin 5mg daily, Aspirin 100mg daily. Symptoms: Chest pain, shortness of breath."
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Run Complete Audit", type="primary", disabled=not text_input):
                    with st.spinner("Running complete audit pipeline..."):
                        result = call_api_audit_text(text_input)
                        if result:
                            st.session_state.audit_result = result
                            st.session_state.uploaded_text = text_input
                            st.success("✅ Audit complete!")
                            st.balloons()
            
            with col2:
                if st.button("📋 View Sample Text"):
                    sample_text = """Patient: Jane Smith, Age: 68
Prescription:
- Warfarin 5 mg once daily
- Aspirin 100 mg once daily
- Metformin 500 mg twice daily

Condition: Atrial fibrillation, Type 2 diabetes

Symptoms: Chest pain, shortness of breath, nausea

Lab values: INR 2.5, HbA1c 7.2%"""
                    st.text_area("Sample Text", value=sample_text, height=150, key="sample_text")
    
    elif page == "View Results":
        st.header("📋 Audit Results")
        
        if st.session_state.audit_result:
            audit_report = st.session_state.audit_result
            
            # Risk score gauge
            if COMPONENTS_LOADED:
                col1, col2 = st.columns([1, 2])
                with col1:
                    risk_score = audit_report.get('risk_score', 0.0)
                    risk_score_gauge(risk_score)
                
                with col2:
                    # Issue distribution
                    critical = len(audit_report.get('critical_issues', []))
                    high = len(audit_report.get('high_issues', []))
                    moderate = len(audit_report.get('moderate_issues', []))
                    safe = len(audit_report.get('safe_items', []))
                    issue_severity_pie(critical, high, moderate, safe)
            else:
                risk_score = audit_report.get('risk_score', 0.0)
                st.metric("Risk Score", f"{risk_score:.2f}/10")
            
            # Agent outputs
            agent_outputs = audit_report.get('agent_outputs', [])
            if agent_outputs:
                if COMPONENTS_LOADED:
                    agent_status_chart(agent_outputs)
                    display_agent_outputs(agent_outputs)
                else:
                    st.subheader("Agent Outputs")
                    for agent in agent_outputs:
                        st.write(f"**{agent.get('agent', 'Unknown')}**: {agent.get('status', 'ok')} - {agent.get('message', '')}")
            
            # Final report
            if COMPONENTS_LOADED:
                display_final_report(audit_report)
            else:
                st.subheader("Audit Report")
                st.json(audit_report)
            
            # Download PDF
            if COMPONENTS_LOADED:
                st.subheader("📥 Download Report")
                try:
                    pdf_bytes = generate_pdf_report(audit_report)
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name="medinsight_audit_report.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Error generating PDF: {e}")
        else:
            st.info("No audit results available. Please run an audit first.")
    
    elif page == "Analytics":
        # Import analytics components
        try:
            from components.analytics import (
                load_analytics_data,
                render_kpi_cards,
                render_risk_score_trend,
                render_severity_distribution_trend,
                render_top_symptoms_chart,
                render_symptom_correlation_heatmap,
                render_critical_case_timeline,
                render_agent_performance_chart,
                render_medication_safety_heatmap,
                render_category_panels,
                render_patient_drawer,
                export_audits_csv,
                explain_chart_with_ai
            )
            
            st.markdown("# 📊 MedInsight Analytics Dashboard")
            st.markdown("---")
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=30),
                    key="analytics_start_date"
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now(),
                    key="analytics_end_date"
                )
            
            severity_filter = st.selectbox(
                "Severity Filter",
                options=["All", "Critical", "High", "Moderate", "Low", "OK"],
                index=0
            )
            
            # Load data
            with st.spinner("Loading analytics data..."):
                start_date_str = start_date.strftime("%Y-%m-%d") if start_date else None
                end_date_str = end_date.strftime("%Y-%m-%d") if end_date else None
                severity_str = severity_filter.lower() if severity_filter != "All" else None
                
                analytics_data = load_analytics_data(
                    API_BASE_URL,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    severity=severity_str
                )
            
            # Export button
            if st.button("📥 Export CSV"):
                if analytics_data.get("audits"):
                    csv_data = export_audits_csv(analytics_data["audits"])
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"medinsight_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            st.markdown("---")
            
            # KPI Cards
            st.markdown("## 📈 Key Performance Indicators")
            kpis = analytics_data.get("kpis", {})
            if kpis:
                render_kpi_cards(kpis)
            else:
                st.warning("No KPI data available. Please ensure audits have been run and saved to MongoDB.")
            st.markdown("---")
            
            # Trend Charts
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Risk Score Trend")
                trends = analytics_data.get("trends", {}).get("trends", [])
                if trends:
                    render_risk_score_trend(trends)
                    if st.button("🤖 Explain this chart", key="explain_risk_trend"):
                        explanation = explain_chart_with_ai("risk_score_trend", {"trends": trends}, API_BASE_URL)
                        st.info(explanation)
                else:
                    st.info("No trend data available")
            
            with col2:
                st.markdown("### Severity Distribution Over Time")
                if trends:
                    render_severity_distribution_trend(trends)
                    if st.button("🤖 Explain this chart", key="explain_severity_trend"):
                        explanation = explain_chart_with_ai("severity_distribution", {"trends": trends}, API_BASE_URL)
                        st.info(explanation)
                else:
                    st.info("No trend data available")
            
            st.markdown("---")
            
            # Symptom Analytics
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Top 10 Symptoms")
                symptom_data = analytics_data.get("symptoms", {})
                if symptom_data:
                    render_top_symptoms_chart(symptom_data)
                    if st.button("🤖 Explain this chart", key="explain_symptoms"):
                        explanation = explain_chart_with_ai("top_symptoms", symptom_data, API_BASE_URL)
                        st.info(explanation)
                else:
                    st.info("No symptom data available")
            
            with col2:
                st.markdown("### Symptom Correlation Heatmap")
                if symptom_data:
                    render_symptom_correlation_heatmap(symptom_data)
                    if st.button("🤖 Explain this chart", key="explain_correlation"):
                        explanation = explain_chart_with_ai("symptom_correlation", symptom_data, API_BASE_URL)
                        st.info(explanation)
                else:
                    st.info("No correlation data available")
            
            st.markdown("---")
            
            # Agent & Medication Analytics
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Agent Performance Distribution")
                agent_data = analytics_data.get("agents", {})
                if agent_data:
                    render_agent_performance_chart(agent_data)
                    if st.button("🤖 Explain this chart", key="explain_agents"):
                        explanation = explain_chart_with_ai("agent_performance", agent_data, API_BASE_URL)
                        st.info(explanation)
                else:
                    st.info("No agent data available")
            
            with col2:
                st.markdown("### Medication Safety Heatmap")
                medication_data = analytics_data.get("medications", {})
                if medication_data:
                    render_medication_safety_heatmap(medication_data)
                    if st.button("🤖 Explain this chart", key="explain_medications"):
                        explanation = explain_chart_with_ai("medication_safety", medication_data, API_BASE_URL)
                        st.info(explanation)
                else:
                    st.info("No medication data available")
            
            st.markdown("---")
            
            # Critical Cases Timeline
            st.markdown("### Critical Cases Timeline")
            audits_data = analytics_data.get("audits", {})
            if audits_data:
                render_critical_case_timeline(audits_data)
                if st.button("🤖 Explain this chart", key="explain_critical"):
                    explanation = explain_chart_with_ai("critical_timeline", audits_data, API_BASE_URL)
                    st.info(explanation)
            else:
                st.info("No audit data available")
            
            st.markdown("---")
            
            # Clinical Category Panels
            st.markdown("## 🏥 Clinical Category Analysis")
            category_data = analytics_data.get("categories", {})
            if category_data:
                render_category_panels(category_data)
            else:
                st.info("No category data available")
            
            st.markdown("---")
            
            # Patient Deep-Dive
            st.markdown("## 👤 Patient Deep-Dive")
            if audits_data and audits_data.get("audits"):
                audits = audits_data["audits"]
                audit_options = {
                    f"Audit {a.get('_id', 'Unknown')[:8]} - Risk: {a.get('risk_score', 0):.2f}": a
                    for a in audits[:50]
                }
                selected_audit_key = st.selectbox(
                    "Select Patient Audit",
                    options=list(audit_options.keys()),
                    index=0
                )
                if selected_audit_key:
                    selected_audit = audit_options[selected_audit_key]
                    render_patient_drawer(selected_audit, API_BASE_URL)
            else:
                st.info("No audit data available for patient deep-dive")
                
        except ImportError as e:
            st.error(f"Error loading analytics dashboard: {e}")
            st.info("Please ensure all analytics components are available.")
        except Exception as e:
            st.error(f"Error rendering analytics: {e}")
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()

