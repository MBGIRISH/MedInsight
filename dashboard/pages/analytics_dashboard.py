"""
Power BI-style Analytics Dashboard for MedInsight.
"""
import streamlit as st
import pandas as pd
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

# Symptom filter (will be populated after data load)
symptom_filter = st.sidebar.multiselect(
    "Symptom Filter",
    options=[],
    default=[]
)

# Category filter
category_filter = st.sidebar.multiselect(
    "Category Filter",
    options=["Respiratory", "Cardiac", "Infection", "Neurological", "Gastrointestinal", "Metabolic"],
    default=[]
)

# ============================================
# MAIN DASHBOARD
# ============================================
st.title("📊 MedInsight Analytics Dashboard")
st.markdown("---")

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

# Export buttons
col1, col2 = st.columns([1, 10])
with col1:
    if st.button("📥 Export CSV"):
        if analytics_data.get("audits"):
            csv_data = export_audits_csv(analytics_data["audits"])
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"medinsight_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

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

