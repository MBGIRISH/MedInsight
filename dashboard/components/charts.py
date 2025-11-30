"""
Chart components for Streamlit dashboard.
"""
import streamlit as st
import pandas as pd
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    st.error("Please install plotly: pip install plotly")
    px = None
    go = None
from typing import List, Dict, Any


def risk_score_gauge(risk_score: float):
    """Display risk score as a gauge chart."""
    if go is None:
        st.error("Plotly not installed")
        return
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score"},
        gauge={
            'axis': {'range': [None, 10]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 3], 'color': "lightgreen"},
                {'range': [3, 6], 'color': "yellow"},
                {'range': [6, 8], 'color': "orange"},
                {'range': [8, 10], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 8
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def agent_status_chart(agent_outputs: List[Dict[str, Any]]):
    """Display agent statuses as a bar chart."""
    if px is None:
        st.error("Plotly not installed")
        return
    agents = [a.get('agent', 'Unknown') for a in agent_outputs]
    scores = [a.get('score', 0.0) for a in agent_outputs]
    statuses = [a.get('status', 'ok') for a in agent_outputs]
    
    # Color mapping
    color_map = {
        'critical': 'red',
        'high': 'orange',
        'moderate': 'yellow',
        'low': 'lightblue',
        'ok': 'green'
    }
    colors = [color_map.get(s.lower(), 'gray') for s in statuses]
    
    df = pd.DataFrame({
        'Agent': agents,
        'Score': scores,
        'Status': statuses
    })
    
    fig = px.bar(
        df,
        x='Agent',
        y='Score',
        color='Status',
        color_discrete_map=color_map,
        title="Agent Analysis Scores",
        labels={'Score': 'Risk Score', 'Agent': 'Agent Name'}
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def issue_severity_pie(critical: int, high: int, moderate: int, safe: int):
    """Display issue distribution as pie chart."""
    if go is None:
        st.error("Plotly not installed")
        return
    labels = ['Critical', 'High', 'Moderate', 'Safe']
    values = [critical, high, moderate, safe]
    colors = ['red', 'orange', 'yellow', 'green']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors)
    )])
    
    fig.update_layout(title="Issue Distribution", height=400)
    st.plotly_chart(fig, use_container_width=True)


def medication_timeline(ner_result: Dict[str, Any]):
    """Display medication timeline from extracted entities."""
    if go is None:
        st.error("Plotly not installed")
        return
    
    if not ner_result or not ner_result.get('entities'):
        st.info("No medication data available for timeline")
        return
    
    entities = ner_result.get('entities', [])
    drugs = [e for e in entities if e.get('type') == 'DRUG']
    dosages = [e for e in entities if e.get('type') == 'DOSAGE']
    frequencies = [e for e in entities if e.get('type') == 'FREQUENCY']
    durations = [e for e in entities if e.get('type') == 'DURATION']
    
    if not drugs:
        st.info("No medications found for timeline")
        return
    
    # Create timeline data
    timeline_data = []
    for i, drug in enumerate(drugs):
        drug_name = drug.get('text', 'Unknown')
        dosage = dosages[i].get('text', '') if i < len(dosages) else ''
        frequency = frequencies[i].get('text', '') if i < len(frequencies) else ''
        duration = durations[i].get('text', '') if i < len(durations) else ''
        
        timeline_data.append({
            'Medication': drug_name,
            'Dosage': dosage,
            'Frequency': frequency,
            'Duration': duration,
            'Position': i
        })
    
    df = pd.DataFrame(timeline_data)
    
    # Create Gantt-like timeline
    fig = go.Figure()
    
    for idx, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Position'], row['Position'] + 1],
            y=[row['Medication'], row['Medication']],
            mode='lines+markers',
            name=row['Medication'],
            line=dict(width=10),
            marker=dict(size=15),
            hovertemplate=f"<b>{row['Medication']}</b><br>" +
                         f"Dosage: {row['Dosage']}<br>" +
                         f"Frequency: {row['Frequency']}<br>" +
                         f"Duration: {row['Duration']}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Medication Timeline",
        xaxis_title="Medication Order",
        yaxis_title="Medication",
        height=300 + len(drugs) * 50,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Also show as table
    st.subheader("Medication Details")
    st.dataframe(df[['Medication', 'Dosage', 'Frequency', 'Duration']], use_container_width=True)


def lab_trend_chart(ner_result: Dict[str, Any]):
    """Display lab value trends from extracted entities."""
    if px is None:
        st.error("Plotly not installed")
        return
    
    if not ner_result or not ner_result.get('entities'):
        st.info("No lab values available for trending")
        return
    
    entities = ner_result.get('entities', [])
    lab_values = [e for e in entities if e.get('type') == 'LAB_VALUE']
    
    if not lab_values:
        st.info("No lab values found")
        return
    
    # Extract test names and values
    test_data = []
    for lv in lab_values:
        text = lv.get('text', '')
        # Try to parse lab value (e.g., "INR 2.5" or "Glucose 95 mg/dL")
        parts = text.split()
        if len(parts) >= 2:
            try:
                test_name = parts[0]
                value = float(parts[1])
                unit = ' '.join(parts[2:]) if len(parts) > 2 else ''
                test_data.append({
                    'Test': test_name,
                    'Value': value,
                    'Unit': unit,
                    'Full Text': text
                })
            except ValueError:
                test_data.append({
                    'Test': text.split()[0] if text.split() else 'Unknown',
                    'Value': 0,
                    'Unit': '',
                    'Full Text': text
                })
    
    if not test_data:
        st.info("No parseable lab values found")
        return
    
    df = pd.DataFrame(test_data)
    
    # Bar chart
    fig = px.bar(
        df,
        x='Test',
        y='Value',
        title="Lab Values Overview",
        labels={'Value': 'Test Value', 'Test': 'Test Name'},
        color='Value',
        color_continuous_scale='RdYlGn_r'
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Line chart for trends (if multiple values)
    if len(df) > 1:
        fig2 = px.line(
            df,
            x='Test',
            y='Value',
            title="Lab Values Trend",
            markers=True
        )
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Table view
    st.subheader("Lab Values Details")
    st.dataframe(df[['Test', 'Value', 'Unit', 'Full Text']], use_container_width=True)

