"""
Power BI-style Analytics Dashboard for MedInsight.
"""
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
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

# Page config
st.set_page_config(
    page_title="MedInsight Analytics",
    page_icon="📊",
    layout="wide"
)

# API endpoint
API_BASE_URL = st.sidebar.text_input(
    "API Base URL",
    value="http://localhost:8000",
    help="Enter the FastAPI backend URL"
)

# ============================================
# FILTERS
# ============================================
st.sidebar.markdown("## 🔍 Filters")

# Date range picker
col1, col2 = st.sidebar.columns(2)
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

# Severity filter
severity_filter = st.sidebar.selectbox(
    "Severity Filter",
    options=["All", "Critical", "High", "Moderate", "Low", "OK"],
    index=0
)

# Category filter
category_filter = st.sidebar.multiselect(
    "Category Filter",
    options=["Respiratory", "Cardiac", "Infection", "Neurological", "Gastrointestinal", "Metabolic"],
    default=[],
    help="Filter by clinical category"
)

# Symptom filter (will be populated after initial data load)
# First, get available symptoms
try:
    symptom_params = {}
    if start_date:
        symptom_params["start_date"] = start_date.strftime("%Y-%m-%d")
    if end_date:
        symptom_params["end_date"] = end_date.strftime("%Y-%m-%d")
    symptom_response = requests.get(f"{API_BASE_URL}/api/analytics/symptoms", params=symptom_params, timeout=5)
    if symptom_response.status_code == 200:
        symptom_data = symptom_response.json()
        available_symptoms = [s.get("symptom", "") for s in symptom_data.get("symptom_frequency", []) if s.get("symptom")]
    else:
        available_symptoms = []
except:
    available_symptoms = []

symptom_filter = st.sidebar.multiselect(
    "Symptom Filter",
    options=available_symptoms[:50] if available_symptoms else [],  # Limit to top 50
    default=[],
    help="Filter by specific symptoms"
)

# ============================================
# MAIN DASHBOARD
# ============================================
# Professional header with styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>📊 MedInsight Analytics Dashboard</h1>
    <p>AI-Powered Clinical Notes Analysis & Risk Assessment</p>
</div>
""", unsafe_allow_html=True)

# Load data with all filters
with st.spinner("Loading analytics data..."):
    start_date_str = start_date.strftime("%Y-%m-%d") if start_date else None
    end_date_str = end_date.strftime("%Y-%m-%d") if end_date else None
    severity_str = severity_filter if severity_filter != "All" else None
    
    analytics_data = load_analytics_data(
        API_BASE_URL,
        start_date=start_date_str,
        end_date=end_date_str,
        severity=severity_str,
        categories=category_filter if category_filter else None,
        symptoms=symptom_filter if symptom_filter else None
    )

# Export buttons - Professional styling
col1, col2, col3 = st.columns([2, 2, 8])
with col1:
    if analytics_data.get("audits") and analytics_data["audits"].get("audits"):
        csv_data = export_audits_csv(analytics_data["audits"])
        st.download_button(
            label="📥 Export CSV",
            data=csv_data,
            file_name=f"medinsight_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
with col2:
    # Filter summary
    active_filters = []
    if severity_filter != "All":
        active_filters.append(f"Severity: {severity_filter}")
    if category_filter:
        active_filters.append(f"Categories: {len(category_filter)}")
    if symptom_filter:
        active_filters.append(f"Symptoms: {len(symptom_filter)}")
    
    if active_filters:
        st.caption(f"Active Filters: {', '.join(active_filters)}")

# ============================================
# KPI CARDS
# ============================================
st.markdown("## 📈 Key Performance Indicators")
kpis = analytics_data.get("kpis", {})
if kpis:
    render_kpi_cards(kpis)
else:
    st.warning("No KPI data available. Please ensure audits have been run and saved to MongoDB.")
st.markdown("---")

# ============================================
# TREND CHARTS
# ============================================
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

# ============================================
# SYMPTOM ANALYTICS
# ============================================
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

# ============================================
# AGENT & MEDICATION ANALYTICS
# ============================================
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

# ============================================
# CRITICAL CASES TIMELINE
# ============================================
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

# ============================================
# CLINICAL CATEGORY PANELS
# ============================================
st.markdown("## 🏥 Clinical Category Analysis")
category_data = analytics_data.get("categories", {})
if category_data:
    render_category_panels(category_data)
else:
    st.info("No category data available")

st.markdown("---")

# ============================================
# PATIENT DEEP-DIVE
# ============================================
st.markdown("## 👤 Patient Deep-Dive")

if audits_data and audits_data.get("audits"):
    audits = audits_data["audits"]
    
    # Patient selector
    audit_options = {
        f"Audit {a.get('_id', 'Unknown')[:8]} - Risk: {a.get('risk_score', 0):.2f}": a
        for a in audits[:50]  # Limit to 50 for performance
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

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>MedInsight Analytics Dashboard | Powered by AI Clinical Notes Auditor</p>
    <p style="font-size: 0.8rem;">Data refreshed automatically. Last updated: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

