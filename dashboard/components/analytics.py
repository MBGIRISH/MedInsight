"""
Power BI-style Analytics Dashboard Components for MedInsight.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import requests
import json
import io

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("Please install plotly: pip install plotly")


# ============================================
# DATA LOADERS
# ============================================

def load_analytics_data(api_base_url: str, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None, severity: Optional[str] = None) -> Dict[str, Any]:
    """Load all analytics data from API."""
    data = {}
    
    try:
        # KPIs
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        response = requests.get(f"{api_base_url}/api/analytics/kpis", params=params, timeout=10)
        if response.status_code == 200:
            data["kpis"] = response.json()
        
        # Trends
        params = {"group_by": "day"}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        response = requests.get(f"{api_base_url}/api/analytics/trends", params=params, timeout=10)
        if response.status_code == 200:
            data["trends"] = response.json()
        
        # Symptoms
        params = {"limit": 20}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        response = requests.get(f"{api_base_url}/api/analytics/symptoms", params=params, timeout=10)
        if response.status_code == 200:
            data["symptoms"] = response.json()
        
        # Agents
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        response = requests.get(f"{api_base_url}/api/analytics/agents", params=params, timeout=10)
        if response.status_code == 200:
            data["agents"] = response.json()
        
        # Categories
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        response = requests.get(f"{api_base_url}/api/analytics/categories", params=params, timeout=10)
        if response.status_code == 200:
            data["categories"] = response.json()
        
        # Medications
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        response = requests.get(f"{api_base_url}/api/analytics/medications", params=params, timeout=10)
        if response.status_code == 200:
            data["medications"] = response.json()
        
        # Audits
        params = {"limit": 1000}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if severity:
            params["severity"] = severity
        response = requests.get(f"{api_base_url}/api/analytics/audits", params=params, timeout=10)
        if response.status_code == 200:
            data["audits"] = response.json()
        
    except Exception as e:
        st.error(f"Error loading analytics data: {e}")
    
    return data


# ============================================
# KPI CARDS
# ============================================

def render_kpi_card(title: str, value: Any, icon: str = "📊", 
                   delta: Optional[str] = None, color: str = "blue"):
    """Render a Power BI-style KPI card."""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: white;
        margin-bottom: 1rem;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;">{title}</div>
                <div style="font-size: 2rem; font-weight: bold;">{icon} {value}</div>
                {f'<div style="font-size: 0.8rem; margin-top: 0.5rem;">{delta}</div>' if delta else ''}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_cards(kpis: Dict[str, Any]):
    """Render all KPI cards."""
    if not kpis:
        st.warning("No KPI data available")
        return
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        render_kpi_card(
            "Total Audits",
            kpis.get("total_audits", 0),
            icon="📋",
            color="blue"
        )
    
    with col2:
        severity_dist = kpis.get("severity_distribution", {})
        critical_count = severity_dist.get("critical", 0)
        render_kpi_card(
            "Critical Cases",
            critical_count,
            icon="🚨",
            color="red"
        )
    
    with col3:
        avg_risk = kpis.get("average_risk_score", 0.0)
        render_kpi_card(
            "Avg Risk Score",
            f"{avg_risk:.2f}/10",
            icon="⚠️",
            color="orange"
        )
    
    with col4:
        top_symptoms = kpis.get("top_symptoms", [])
        top_symptom = top_symptoms[0]["symptom"] if top_symptoms else "None"
        render_kpi_card(
            "Top Symptom",
            top_symptom[:20] + "..." if len(top_symptom) > 20 else top_symptom,
            icon="🤒",
            color="purple"
        )
    
    with col5:
        top_agent = kpis.get("most_triggered_agent", "None")
        render_kpi_card(
            "Top Agent",
            top_agent,
            icon="🤖",
            color="green"
        )


# ============================================
# CHARTS
# ============================================

def render_risk_score_trend(trends_data: List[Dict[str, Any]]):
    """Render risk score trend line chart."""
    if not PLOTLY_AVAILABLE or not trends_data:
        st.info("No trend data available")
        return
    
    df = pd.DataFrame(trends_data)
    df["date"] = df["_id"]
    df["avg_risk_score"] = df["avg_risk_score"].fillna(0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["avg_risk_score"],
        mode='lines+markers',
        name='Average Risk Score',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Risk Score Trend Over Time",
        xaxis_title="Date",
        yaxis_title="Average Risk Score",
        height=400,
        template="plotly_white",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_severity_distribution_trend(trends_data: List[Dict[str, Any]]):
    """Render severity distribution stacked area chart."""
    if not PLOTLY_AVAILABLE or not trends_data:
        st.info("No trend data available")
        return
    
    df = pd.DataFrame(trends_data)
    df["date"] = df["_id"]
    
    fig = go.Figure()
    
    colors = {
        "critical": "#dc3545",
        "high": "#fd7e14",
        "moderate": "#ffc107",
        "low": "#17a2b8",
        "ok": "#28a745"
    }
    
    for severity in ["critical", "high", "moderate", "low", "ok"]:
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df[severity],
            mode='lines',
            name=severity.upper(),
            stackgroup='one',
            fillcolor=colors[severity],
            line=dict(width=0.5, color=colors[severity])
        ))
    
    fig.update_layout(
        title="Severity Distribution Over Time",
        xaxis_title="Date",
        yaxis_title="Number of Cases",
        height=400,
        template="plotly_white",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_top_symptoms_chart(symptom_data: Dict[str, Any]):
    """Render top symptoms bar chart."""
    if not PLOTLY_AVAILABLE or not symptom_data:
        st.info("No symptom data available")
        return
    
    symptoms = symptom_data.get("symptom_frequency", [])
    if not symptoms:
        st.info("No symptoms found")
        return
    
    df = pd.DataFrame(symptoms[:10])
    
    fig = px.bar(
        df,
        x='count',
        y='symptom',
        orientation='h',
        title="Top 10 Symptoms",
        labels={'count': 'Frequency', 'symptom': 'Symptom'},
        color='count',
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        height=500,
        template="plotly_white",
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_symptom_correlation_heatmap(symptom_data: Dict[str, Any]):
    """Render symptom correlation heatmap."""
    if not PLOTLY_AVAILABLE or not symptom_data:
        st.info("No correlation data available")
        return
    
    correlations = symptom_data.get("correlations", [])
    if not correlations:
        st.info("No correlation data available. Run more audits to see symptom correlations.")
        return
    
    # Build correlation matrix
    symptoms = list(set([c.get("symptom1") for c in correlations if c.get("symptom1")] + 
                        [c.get("symptom2") for c in correlations if c.get("symptom2")]))
    symptoms = sorted(symptoms)[:10]  # Limit to top 10
    
    if not symptoms:
        st.info("No symptoms found for correlation analysis")
        return
    
    matrix = np.zeros((len(symptoms), len(symptoms)))
    for corr in correlations:
        sym1 = corr.get("symptom1")
        sym2 = corr.get("symptom2")
        if sym1 in symptoms and sym2 in symptoms:
            i = symptoms.index(sym1)
            j = symptoms.index(sym2)
            co_occurrence = corr.get("co_occurrence", 0)
            matrix[i][j] = co_occurrence
            matrix[j][i] = co_occurrence
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=symptoms,
        y=symptoms,
        colorscale='YlOrRd',
        text=matrix.astype(int),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{x} & %{y}<br>Co-occurrence: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Symptom Correlation Heatmap",
        height=500,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_critical_case_timeline(audits_data: Dict[str, Any]):
    """Render critical case timeline bar chart."""
    if not PLOTLY_AVAILABLE or not audits_data:
        st.info("No audit data available")
        return
    
    audits = audits_data.get("audits", [])
    critical_audits = [a for a in audits if a.get("risk_score", 0) >= 9.0]
    
    if not critical_audits:
        st.info("No critical cases found")
        return
    
    # Group by date
    dates = {}
    for audit in critical_audits:
        date_str = audit.get("created_at", "")[:10] if audit.get("created_at") else "Unknown"
        dates[date_str] = dates.get(date_str, 0) + 1
    
    df = pd.DataFrame(list(dates.items()), columns=["Date", "Count"])
    df = df.sort_values("Date")
    
    fig = px.bar(
        df,
        x="Date",
        y="Count",
        title="Critical Cases Timeline",
        labels={'Count': 'Number of Critical Cases', 'Date': 'Date'},
        color='Count',
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        height=400,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_agent_performance_chart(agent_data: Dict[str, Any]):
    """Render agent performance distribution chart."""
    if not PLOTLY_AVAILABLE or not agent_data:
        st.info("No agent data available")
        return
    
    agents = agent_data.get("agent_performance", [])
    if not agents:
        st.info("No agent data available")
        return
    
    df = pd.DataFrame(agents)
    
    fig = go.Figure()
    
    for agent in agents:
        breakdown = agent["severity_breakdown"]
        fig.add_trace(go.Bar(
            name=agent["agent"],
            x=[agent["agent"]],
            y=[breakdown["critical"]],
            marker_color='#dc3545',
            legendgroup='critical'
        ))
        fig.add_trace(go.Bar(
            name=agent["agent"],
            x=[agent["agent"]],
            y=[breakdown["high"]],
            marker_color='#fd7e14',
            legendgroup='high'
        ))
        fig.add_trace(go.Bar(
            name=agent["agent"],
            x=[agent["agent"]],
            y=[breakdown["moderate"]],
            marker_color='#ffc107',
            legendgroup='moderate'
        ))
        fig.add_trace(go.Bar(
            name=agent["agent"],
            x=[agent["agent"]],
            y=[breakdown["low"]],
            marker_color='#17a2b8',
            legendgroup='low'
        ))
    
    fig.update_layout(
        title="Agent Performance Distribution",
        xaxis_title="Agent",
        yaxis_title="Number of Triggers",
        barmode='stack',
        height=400,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_medication_safety_heatmap(medication_data: Dict[str, Any]):
    """Render medication safety heatmap."""
    if not PLOTLY_AVAILABLE or not medication_data:
        st.info("No medication data available")
        return
    
    dosage_issues = medication_data.get("common_dosage_issues", [])
    interaction_issues = medication_data.get("common_interactions", [])
    
    if not dosage_issues and not interaction_issues:
        st.info("No medication safety issues found")
        return
    
    # Create heatmap data
    issues = []
    for issue in dosage_issues[:5]:
        issues.append({"Type": "Dosage Error", "Issue": issue["issue"][:50], "Count": issue["count"]})
    for issue in interaction_issues[:5]:
        issues.append({"Type": "Interaction", "Issue": issue["interaction"][:50], "Count": issue["count"]})
    
    if not issues:
        st.info("No medication safety data available")
        return
    
    df = pd.DataFrame(issues)
    
    # Pivot for heatmap
    pivot_df = df.pivot_table(values='Count', index='Issue', columns='Type', fill_value=0)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale='Reds',
        text=pivot_df.values,
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Medication Safety Heatmap",
        height=400,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ============================================
# CLINICAL CATEGORY PANELS
# ============================================

def render_category_panel(category_name: str, category_data: Dict[str, Any]):
    """Render a clinical category panel with proper alignment."""
    # Handle different data formats from API
    if isinstance(category_data, dict):
        count = category_data.get("total_cases", category_data.get("count", 0))
        severity = category_data.get("severity_breakdown", {})
        avg_risk = category_data.get("average_risk_score", category_data.get("avg_risk_score", 0.0))
    else:
        count = 0
        severity = {}
        avg_risk = 0.0
    
    # Ensure severity breakdown is a dict
    if not isinstance(severity, dict):
        severity = {}
    
    st.markdown(f"""
    <div style="
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        height: 100%;
        display: flex;
        flex-direction: column;
        min-height: 180px;
    ">
        <h3 style="margin-top: 0; margin-bottom: 1rem; color: #667eea; font-size: 1.2rem; font-weight: 600;">{category_name}</h3>
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-top: 0.5rem; margin-bottom: 1rem;">
            <div style="flex: 1;">
                <div style="font-size: 2rem; font-weight: bold; color: #333; line-height: 1.2;">{count}</div>
                <div style="color: #666; font-size: 0.9rem; margin-top: 0.25rem;">Total Cases</div>
            </div>
            <div style="flex: 1; text-align: right;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #333; line-height: 1.2;">{avg_risk:.2f}</div>
                <div style="color: #666; font-size: 0.9rem; margin-top: 0.25rem;">Avg Risk</div>
            </div>
        </div>
        <div style="margin-top: auto; display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center;">
            <span style="background: #dc3545; color: white; padding: 0.3rem 0.6rem; border-radius: 5px; font-size: 0.8rem; white-space: nowrap; display: inline-block;">
                Critical: {severity.get('critical', 0)}
            </span>
            <span style="background: #fd7e14; color: white; padding: 0.3rem 0.6rem; border-radius: 5px; font-size: 0.8rem; white-space: nowrap; display: inline-block;">
                High: {severity.get('high', 0)}
            </span>
            <span style="background: #ffc107; color: white; padding: 0.3rem 0.6rem; border-radius: 5px; font-size: 0.8rem; white-space: nowrap; display: inline-block;">
                Moderate: {severity.get('moderate', 0)}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_category_panels(category_data: Any):
    """Render all clinical category panels with proper grid alignment."""
    if not category_data:
        st.info("No category data available")
        return
    
    # Handle different data formats from API
    categories_dict = {}
    
    if isinstance(category_data, list):
        # API returns a list of category objects
        for cat in category_data:
            cat_name = cat.get("category", "Unknown")
            categories_dict[cat_name] = {
                "total_cases": cat.get("total_cases", 0),
                "average_risk_score": cat.get("average_risk_score", 0.0),
                "severity_breakdown": cat.get("severity_breakdown", {})
            }
    elif isinstance(category_data, dict):
        # Check if it's a dict with "categories" key
        if "categories" in category_data:
            categories_dict = category_data["categories"]
        else:
            # Direct dict of categories
            categories_dict = category_data
    
    if not categories_dict:
        st.info("No clinical categories found")
        return
    
    # Convert to list for consistent processing
    category_list = list(categories_dict.items()) if isinstance(categories_dict, dict) else []
    
    if not category_list:
        st.info("No clinical categories to display")
        return
    
    # Render in rows of 3 for proper grid alignment
    for row_start in range(0, len(category_list), 3):
        cols = st.columns(3)
        row_items = category_list[row_start:row_start + 3]
        
        for col_idx, (category_name, category_info) in enumerate(row_items):
            with cols[col_idx]:
                render_category_panel(category_name, category_info)


# ============================================
# PATIENT DEEP-DIVE DRAWER
# ============================================

def render_patient_drawer(audit: Dict[str, Any], api_base_url: str):
    """Render patient deep-dive drawer."""
    with st.expander(f"🔍 Patient Deep-Dive: Audit {audit.get('_id', 'Unknown')[:8]}", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Risk Score")
            risk_score = audit.get("risk_score", 0.0)
            st.metric("Risk Score", f"{risk_score:.2f}/10")
            
            st.markdown("### Symptoms")
            symptoms = audit.get("ner_result", {}).get("normalized_entities", {}).get("symptoms", [])
            if symptoms:
                for symptom in symptoms[:10]:
                    st.markdown(f"• {symptom}")
            else:
                st.info("No symptoms detected")
        
        with col2:
            st.markdown("### Vital Signs")
            vitals = audit.get("ner_result", {}).get("normalized_entities", {}).get("vitals", [])
            if vitals:
                for vital in vitals[:5]:
                    if isinstance(vital, dict):
                        st.markdown(f"• {vital.get('type', 'Unknown')}: {vital.get('original', 'N/A')}")
                    else:
                        st.markdown(f"• {vital}")
            else:
                st.info("No vitals detected")
            
            st.markdown("### Red Flags")
            critical_issues = audit.get("critical_issues", [])
            if critical_issues:
                for issue in critical_issues[:3]:
                    st.error(f"🚨 {issue.get('message', 'Unknown issue')}")
            else:
                st.success("No critical red flags")
        
        st.markdown("### Recommendations")
        recommendations = audit.get("recommendations", [])
        if recommendations:
            for i, rec in enumerate(recommendations[:5], 1):
                st.markdown(f"{i}. {rec}")
        else:
            st.info("No recommendations available")
        
        # Download PDF button
        try:
            from components.report_viewer import generate_pdf_report
            pdf_bytes = generate_pdf_report(audit)
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"audit_{audit.get('_id', 'unknown')[:8]}.pdf",
                mime="application/pdf",
                key=f"pdf_download_{audit.get('_id')}"
            )
        except Exception as e:
            st.error(f"Error generating PDF: {e}")
            st.caption("PDF generation requires reportlab. Install with: pip install reportlab")


# ============================================
# EXPORT FUNCTIONS
# ============================================

def export_audits_csv(audits_data: Dict[str, Any]) -> bytes:
    """Export audits as CSV."""
    if isinstance(audits_data, dict):
        audits = audits_data.get("audits", [])
    elif isinstance(audits_data, list):
        audits = audits_data
    else:
        audits = []
    
    if not audits:
        return b""
    
    # Flatten audit data
    rows = []
    for audit in audits:
        symptoms = audit.get("ner_result", {}).get("normalized_entities", {}).get("symptoms", [])
        symptoms_str = ", ".join([str(s) for s in symptoms[:5]]) if symptoms else "None"
        
        row = {
            "audit_id": audit.get("_id", ""),
            "created_at": audit.get("created_at", ""),
            "risk_score": audit.get("risk_score", 0.0),
            "critical_issues": len(audit.get("critical_issues", [])),
            "high_issues": len(audit.get("high_issues", [])),
            "moderate_issues": len(audit.get("moderate_issues", [])),
            "symptoms": symptoms_str
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode('utf-8')


def export_analytics_pdf(analytics_data: Dict[str, Any]) -> bytes:
    """Export analytics dashboard as PDF (placeholder - would need reportlab)."""
    # This would generate a PDF report of the analytics
    # For now, return a simple text representation
    return json.dumps(analytics_data, indent=2).encode('utf-8')


# ============================================
# AI EXPLANATION
# ============================================

def explain_chart_with_ai(chart_type: str, data: Dict[str, Any], api_base_url: str) -> str:
    """Use LLM to explain a chart."""
    try:
        # This would call an LLM service to explain the chart
        # For now, return a simple explanation
        explanation = f"This {chart_type} chart shows the following insights:\n\n"
        
        if chart_type == "risk_score_trend":
            explanation += "The risk score trend indicates how the average risk score has changed over time. "
            explanation += "Higher values indicate more critical cases, while lower values suggest safer assessments."
        elif chart_type == "severity_distribution":
            explanation += "The severity distribution shows the breakdown of cases by severity level over time. "
            explanation += "This helps identify patterns in case severity and potential trends."
        elif chart_type == "top_symptoms":
            explanation += "This chart displays the most frequently occurring symptoms in the dataset. "
            explanation += "This can help identify common health concerns and patterns."
        
        return explanation
    except Exception as e:
        return f"Error generating explanation: {e}"

