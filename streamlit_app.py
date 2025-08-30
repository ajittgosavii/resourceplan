import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta, date
import json
import logging
import hashlib
import uuid
import time
from collections import defaultdict
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Cloud Operations Resource Planning - 5 Year Strategic Plan",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.5rem;
    }
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def load_teams_and_categories():
    """Load simplified teams and categories focused on Back Office and Database Operations"""
    teams = {
        'BCO': {
            'name': 'Back Office Cloud Operations Team',
            'current_size': 6,
            'location': 'India',
            'manager': 'Sarah Johnson',
            'specializations': ['Cloud Operations', 'Monitoring', 'Support']
        },
        'DBO': {
            'name': 'Database Operations Team',
            'current_size': 7,
            'location': 'USA',
            'manager': 'Robert Wilson',
            'specializations': ['Database Management', 'Performance Tuning', 'Backup & Recovery']
        }
    }
    
    # Activity categories relevant to both teams
    categories = {
        'AWS Infrastructure Management': {
            'activities': 40,
            'automation_potential': 0.85,
            'complexity': 'High',
            'priority': 'P1',
            'current_maturity': 3.2,
            'target_maturity': 4.8,
        },
        'Database Operations': {
            'activities': 45,
            'automation_potential': 0.75,
            'complexity': 'High',
            'priority': 'P1',
            'current_maturity': 3.5,
            'target_maturity': 4.6,
        },
        'Monitoring & Alerting': {
            'activities': 35,
            'automation_potential': 0.85,
            'complexity': 'High',
            'priority': 'P0',
            'current_maturity': 3.7,
            'target_maturity': 4.8,
        },
        'Incident Management': {
            'activities': 24,
            'automation_potential': 0.60,
            'complexity': 'High',
            'priority': 'P0',
            'current_maturity': 3.5,
            'target_maturity': 4.6,
        },
        'Change Management': {
            'activities': 20,
            'automation_potential': 0.50,
            'complexity': 'Medium',
            'priority': 'P2',
            'current_maturity': 3.0,
            'target_maturity': 4.2,
        },
        'Data Management & Backup': {
            'activities': 20,
            'automation_potential': 0.90,
            'complexity': 'Medium',
            'priority': 'P1',
            'current_maturity': 3.4,
            'target_maturity': 4.7,
        }
    }
    
    return teams, categories

def calculate_financial_model(bco_size, dbo_size, investment_params, timeline_years=5):
    """Calculate simplified financial model"""
    
    # Extract parameters
    avg_fte_cost = investment_params['avg_fte_cost']
    automation_investment = investment_params['automation_investment']
    training_budget = investment_params['training_budget']
    
    # Calculate year-over-year financial model
    financial_model = []
    cumulative_investment = 0
    cumulative_savings = 0
    
    total_current_fte = bco_size + dbo_size
    
    for year in range(timeline_years):
        # Annual investment
        if year == 0:
            annual_investment = automation_investment + training_budget
        else:
            annual_investment = training_budget * (1.03 ** year)  # 3% inflation
        
        # Savings calculation
        # 1. Direct FTE cost avoidance through automation
        base_team_growth = total_current_fte * (1.12 ** year)  # 12% annual growth without automation
        automation_fte_reduction = base_team_growth * (0.05 + year * 0.06)  # Progressive automation
        fte_cost_savings = automation_fte_reduction * avg_fte_cost
        
        # 2. Operational efficiency gains
        efficiency_multiplier = 1 + (year * 0.12)  # 12% annual efficiency improvement
        operational_savings = 150 * 1000 * efficiency_multiplier  # Base operational savings
        
        # 3. Risk avoidance and cost optimization
        incident_cost_avoidance = (80 + year * 20) * 1000  # Avoided incident costs
        
        total_savings = fte_cost_savings + operational_savings + incident_cost_avoidance
        
        # Calculate financial metrics
        net_benefit = total_savings - annual_investment
        
        cumulative_investment += annual_investment
        cumulative_savings += total_savings
        
        financial_model.append({
            'Year': 2025 + year,
            'Annual_Investment': annual_investment,
            'Total_Savings': total_savings,
            'Net_Benefit': net_benefit,
            'Cumulative_Investment': cumulative_investment,
            'Cumulative_Savings': cumulative_savings,
            'Cumulative_Net': cumulative_savings - cumulative_investment,
            'ROI_Percentage': (cumulative_savings / cumulative_investment - 1) * 100 if cumulative_investment > 0 else 0
        })
    
    return pd.DataFrame(financial_model)

# Initialize session state
if 'teams' not in st.session_state:
    teams, categories = load_teams_and_categories()
    st.session_state.teams = teams
    st.session_state.categories = categories

# Main application header
st.markdown('<div class="main-header"><h1>☁️ Cloud Operations 5-Year Strategic Resource Plan</h1><p>Back Office Cloud Operations & Database Operations Planning</p></div>', unsafe_allow_html=True)

# Tab-based navigation
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive Dashboard",
    "👥 Resource Planning", 
    "💰 Financial Analysis",
    "📈 Analytics",
    "🤖 Automation Strategy",
    "📋 Implementation Plan"
])

with tab1:
    st.header("Executive Strategic Overview")
    
    teams = st.session_state.teams
    categories = st.session_state.categories
    
    # Calculate current metrics
    current_total_fte = sum([team_data['current_size'] for team_data in teams.values()])
    total_activities = sum([cat_data['activities'] for cat_data in categories.values()])
    avg_automation_potential = np.mean([cat_data['automation_potential'] for cat_data in categories.values()])
    
    # Executive KPI dashboard
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Current Total FTE", current_total_fte)
    
    with col2:
        projected_baseline = int(current_total_fte * 1.68)  # 68% growth over 5 years
        st.metric("Year 5 Baseline Growth", projected_baseline, f"+{projected_baseline - current_total_fte}")
    
    with col3:
        projected_optimized = int(current_total_fte * 1.25)  # 25% growth with automation
        st.metric("Year 5 Optimized", projected_optimized, f"+{projected_optimized - current_total_fte}")
    
    with col4:
        fte_avoidance = projected_baseline - projected_optimized
        st.metric("FTE Avoidance", fte_avoidance)
    
    with col5:
        cost_avoidance_5yr = fte_avoidance * 130 * 5  # $130K average cost per FTE
        st.metric("5-Year Cost Avoidance", f"${cost_avoidance_5yr/1000:.1f}M")
    
    with col6:
        productivity_gain = avg_automation_potential * 100
        st.metric("Automation Potential", f"{productivity_gain:.0f}%")
    
    st.markdown("---")
    
    # Team overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Team Overview")
        
        team_data = []
        for team_code, team_info in teams.items():
            team_data.append({
                'Team': team_code,
                'Name': team_info['name'],
                'Current Size': team_info['current_size'],
                'Location': team_info['location'],
                'Manager': team_info['manager']
            })
        
        team_df = pd.DataFrame(team_data)
        st.dataframe(team_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Location Distribution")
        
        location_fte = defaultdict(int)
        for team_code, team_data in teams.items():
            location = team_data['location']
            location_fte[location] += team_data['current_size']
        
        fig = px.pie(
            values=list(location_fte.values()),
            names=list(location_fte.keys()),
            title="FTE Distribution by Location"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Activity analysis
    st.subheader("Activity Category Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        category_activities = [cat_data['activities'] for cat_data in categories.values()]
        category_names = list(categories.keys())
        
        fig = px.bar(
            x=category_names,
            y=category_activities,
            title=f"Distribution of {total_activities} Total Activities",
            color=category_activities,
            color_continuous_scale="Viridis"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        automation_potential = [cat_data['automation_potential'] for cat_data in categories.values()]
        
        fig = px.bar(
            x=category_names,
            y=automation_potential,
            title="Automation Potential by Category",
            color=automation_potential,
            color_continuous_scale="RdYlGn"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Resource Planning & Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Team Sizes")
        
        # Allow user to input team sizes
        bco_size = st.number_input(
            "Back Office Cloud Operations Team Size", 
            min_value=1, max_value=50, 
            value=st.session_state.teams['BCO']['current_size']
        )
        
        dbo_size = st.number_input(
            "Database Operations Team Size", 
            min_value=1, max_value=50, 
            value=st.session_state.teams['DBO']['current_size']
        )
        
        # Update session state
        st.session_state.teams['BCO']['current_size'] = bco_size
        st.session_state.teams['DBO']['current_size'] = dbo_size
        
        total_fte = bco_size + dbo_size
        st.metric("Total Current FTE", total_fte)
    
    with col2:
        st.subheader("Team Locations")
        
        st.info(f"**Back Office Cloud Operations**: India ({bco_size} FTE)")
        st.info(f"**Database Operations**: USA ({dbo_size} FTE)")
        
        # Location distribution chart
        fig = px.pie(
            values=[bco_size, dbo_size],
            names=['India (BCO)', 'USA (DBO)'],
            title="Resource Distribution by Location"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Resource growth projections
    st.subheader("5-Year Resource Growth Projections")
    
    growth_scenarios = {
        'Conservative (8% annually)': 0.08,
        'Base Case (12% annually)': 0.12,
        'Optimistic (18% annually)': 0.18
    }
    
    years = list(range(2025, 2030))
    projection_data = []
    
    for scenario, growth_rate in growth_scenarios.items():
        bco_projections = [bco_size * ((1 + growth_rate) ** i) for i in range(5)]
        dbo_projections = [dbo_size * ((1 + growth_rate) ** i) for i in range(5)]
        total_projections = [bco + dbo for bco, dbo in zip(bco_projections, dbo_projections)]
        
        for i, year in enumerate(years):
            projection_data.append({
                'Year': year,
                'Scenario': scenario,
                'BCO': int(bco_projections[i]),
                'DBO': int(dbo_projections[i]),
                'Total': int(total_projections[i])
            })
    
    projection_df = pd.DataFrame(projection_data)
    
    # Visualization
    fig = px.line(
        projection_df, 
        x='Year', 
        y='Total', 
        color='Scenario',
        title='5-Year Resource Growth Scenarios',
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed projections table
    st.subheader("Detailed Resource Projections")
    
    # Pivot table for better display
    pivot_df = projection_df.pivot(index='Year', columns='Scenario', values='Total')
    st.dataframe(pivot_df, use_container_width=True)

with tab3:
    st.header("Financial Analysis & ROI Modeling")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Investment Parameters")
        
        investment_params = {
            'avg_fte_cost': st.number_input("Average FTE Cost ($K)", 80, 200, 130),
            'automation_investment': st.number_input("Automation Investment ($K)", 100, 1000, 450),
            'training_budget': st.number_input("Annual Training Budget ($K)", 20, 200, 85)
        }
    
    with col2:
        st.subheader("Current Configuration")
        
        current_bco = st.session_state.teams['BCO']['current_size']
        current_dbo = st.session_state.teams['DBO']['current_size']
        
        st.metric("BCO Team Size", current_bco)
        st.metric("DBO Team Size", current_dbo)
        st.metric("Total FTE", current_bco + current_dbo)
    
    if st.button("Generate Financial Model", type="primary"):
        with st.spinner("Calculating financial projections..."):
            financial_model = calculate_financial_model(
                current_bco, current_dbo, investment_params
            )
            st.session_state.financial_model = financial_model
            st.success("Financial model generated successfully!")
    
    # Display financial results
    if 'financial_model' in st.session_state:
        st.subheader("5-Year Financial Analysis")
        
        financial_df = st.session_state.financial_model
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_investment = financial_df['Cumulative_Investment'].iloc[-1]
        total_savings = financial_df['Cumulative_Savings'].iloc[-1]
        final_roi = financial_df['ROI_Percentage'].iloc[-1]
        net_value = financial_df['Cumulative_Net'].iloc[-1]
        
        with col1:
            st.metric("Total Investment", f"${total_investment/1000:.1f}M")
        with col2:
            st.metric("Total Savings", f"${total_savings/1000:.1f}M")
        with col3:
            st.metric("Net Value", f"${net_value/1000:.1f}M")
        with col4:
            st.metric("5-Year ROI", f"{final_roi:.1f}%")
        
        # Financial chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=financial_df['Year'],
            y=financial_df['Cumulative_Investment'],
            mode='lines+markers',
            name='Cumulative Investment',
            line=dict(color='red', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=financial_df['Year'],
            y=financial_df['Cumulative_Savings'],
            mode='lines+markers',
            name='Cumulative Savings',
            line=dict(color='green', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=financial_df['Year'],
            y=financial_df['Cumulative_Net'],
            mode='lines+markers',
            name='Net Cumulative Value',
            line=dict(color='blue', width=3)
        ))
        
        fig.update_layout(
            title="5-Year Financial Model",
            xaxis_title="Year",
            yaxis_title="Value ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed breakdown
        st.subheader("Year-by-Year Breakdown")
        
        display_df = financial_df.copy()
        for col in ['Annual_Investment', 'Total_Savings', 'Cumulative_Investment', 'Cumulative_Savings']:
            display_df[col] = display_df[col].apply(lambda x: f"${x/1000:.1f}M")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)

with tab4:
    st.header("Analytics & Performance Insights")
    
    # Generate sample performance data
    current_date = datetime.now()
    dates = pd.date_range(start=current_date - timedelta(days=90), end=current_date, freq='D')
    
    performance_data = pd.DataFrame({
        'Date': dates,
        'Incident_Count': np.random.poisson(2.5, len(dates)),
        'MTTR_Minutes': np.random.normal(35, 6, len(dates)),
        'Availability_Percent': np.random.normal(99.85, 0.1, len(dates)),
        'Cost_Per_Day': np.random.normal(6500, 350, len(dates))
    })
    
    # KPIs
    st.subheader("Operational KPIs (Last 7 Days)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_incidents = performance_data['Incident_Count'].tail(7).mean()
        st.metric("Avg Daily Incidents", f"{avg_incidents:.1f}")
    
    with col2:
        avg_mttr = performance_data['MTTR_Minutes'].tail(7).mean()
        st.metric("Avg MTTR", f"{avg_mttr:.0f} min")
    
    with col3:
        avg_availability = performance_data['Availability_Percent'].tail(7).mean()
        st.metric("Availability", f"{avg_availability:.2f}%")
    
    with col4:
        avg_cost = performance_data['Cost_Per_Day'].tail(7).mean()
        st.metric("Daily OpEx", f"${avg_cost:.0f}")
    
    # Trends
    st.subheader("90-Day Performance Trends")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Daily Incidents', 'MTTR Trends', 'Availability %', 'Daily Costs')
    )
    
    fig.add_trace(go.Scatter(x=performance_data['Date'], y=performance_data['Incident_Count'], 
                           name='Incidents'), row=1, col=1)
    fig.add_trace(go.Scatter(x=performance_data['Date'], y=performance_data['MTTR_Minutes'], 
                           name='MTTR'), row=1, col=2)
    fig.add_trace(go.Scatter(x=performance_data['Date'], y=performance_data['Availability_Percent'], 
                           name='Availability'), row=2, col=1)
    fig.add_trace(go.Scatter(x=performance_data['Date'], y=performance_data['Cost_Per_Day'], 
                           name='Daily Cost'), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Team performance comparison
    st.subheader("Team Performance Analysis")
    
    team_performance = {
        'BCO Team (India)': {
            'Incidents_Handled': 145,
            'Avg_Resolution_Time': 32,
            'Availability_Target': 99.9,
            'Cost_Efficiency': 85
        },
        'DBO Team (USA)': {
            'Incidents_Handled': 89,
            'Avg_Resolution_Time': 28,
            'Availability_Target': 99.95,
            'Cost_Efficiency': 92
        }
    }
    
    perf_df = pd.DataFrame(team_performance).T.reset_index()
    perf_df.rename(columns={'index': 'Team'}, inplace=True)
    
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

with tab5:
    st.header("Automation Strategy")
    
    categories = st.session_state.categories
    
    # Automation opportunity matrix
    st.subheader("Automation Opportunities")
    
    complexity_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
    
    automation_data = []
    for cat_name, cat_data in categories.items():
        effort_score = complexity_mapping.get(cat_data['complexity'], 2)
        impact_score = cat_data['automation_potential'] * 10
        roi_score = (impact_score / effort_score) * cat_data['activities'] / 10
        
        automation_data.append({
            'Category': cat_name,
            'Activities': cat_data['activities'],
            'Automation_Potential': f"{cat_data['automation_potential']*100:.0f}%",
            'Complexity': cat_data['complexity'],
            'Effort_Score': effort_score,
            'Impact_Score': impact_score,
            'ROI_Score': roi_score
        })
    
    automation_df = pd.DataFrame(automation_data)
    
    # Bubble chart
    fig = px.scatter(
        automation_df,
        x='Effort_Score',
        y='Impact_Score',
        size='Activities',
        color='ROI_Score',
        hover_name='Category',
        title="Automation Opportunity Matrix (Impact vs Effort)",
        color_continuous_scale="RdYlGn"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Automation priorities
    st.subheader("Top Automation Priorities")
    
    top_automation = automation_df.nlargest(3, 'ROI_Score')
    
    for i, row in top_automation.iterrows():
        with st.expander(f"Priority {row.name + 1}: {row['Category']}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Activities", row['Activities'])
                st.metric("Complexity", row['Complexity'])
            
            with col2:
                st.metric("Automation Potential", row['Automation_Potential'])
                st.metric("Effort Score", f"{row['Effort_Score']}/4")
            
            with col3:
                st.metric("Impact Score", f"{row['Impact_Score']:.1f}/10")
                st.metric("ROI Score", f"{row['ROI_Score']:.1f}")
    
    # Implementation timeline
    st.subheader("Suggested Implementation Timeline")
    
    timeline_data = {
        'Phase 1 (Q1-Q2 2025)': ['Data Management & Backup', 'Monitoring & Alerting'],
        'Phase 2 (Q3-Q4 2025)': ['AWS Infrastructure Management', 'Database Operations'],
        'Phase 3 (Q1-Q2 2026)': ['Incident Management', 'Change Management']
    }
    
    for phase, activities in timeline_data.items():
        st.write(f"**{phase}**")
        for activity in activities:
            st.write(f"• {activity}")

with tab6:
    st.header("Implementation Plan & Roadmap")
    
    # Implementation phases
    st.subheader("5-Year Implementation Roadmap")
    
    # Year 1 (2025)
    with st.expander("Year 1 (2025) - Foundation & Quick Wins", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Q1-Q2 Initiatives:**")
            st.markdown("• Implement automated backup systems")
            st.markdown("• Deploy advanced monitoring dashboards")
            st.markdown("• Begin team skills assessment")
            st.markdown("• Establish automation governance")
            
        with col2:
            st.markdown("**Q3-Q4 Initiatives:**")
            st.markdown("• AWS infrastructure automation")
            st.markdown("• Database performance optimization")
            st.markdown("• Cross-team training programs")
            st.markdown("• ROI measurement framework")
    
    # Year 2-3 (2026-2027)
    with st.expander("Years 2-3 (2026-2027) - Scale & Optimize"):
        st.markdown("**Key Focus Areas:**")
        st.markdown("• Advanced automation deployment")
        st.markdown("• AI-powered incident management")
        st.markdown("• Cross-location collaboration optimization")
        st.markdown("• Comprehensive skills development")
        st.markdown("• Process standardization")
    
    # Year 4-5 (2028-2029)
    with st.expander("Years 4-5 (2028-2029) - Innovation & Excellence"):
        st.markdown("**Strategic Initiatives:**")
        st.markdown("• Full automation maturity")
        st.markdown("• Predictive operations capabilities")
        st.markdown("• Industry-leading practices adoption")
        st.markdown("• Knowledge sharing ecosystem")
        st.markdown("• Continuous improvement culture")
    
    # Success metrics
    st.subheader("Success Metrics & KPIs")
    
    success_metrics = {
        'Operational Excellence': [
            'System Availability: 99.9% → 99.95%',
            'MTTR: 35 min → 15 min',
            'Incident Volume: -40%'
        ],
        'Automation Achievement': [
            'Automated Tasks: 45% → 85%',
            'Manual Effort Reduction: 60%',
            'Process Efficiency: +75%'
        ],
        'Team Development': [
            'Skills Proficiency: +40%',
            'Cross-training Coverage: 90%',
            'Knowledge Sharing Score: 4.5/5'
        ],
        'Financial Performance': [
            f'Cost Avoidance: ${(13 * 130 * 5)/1000:.1f}M over 5 years',
            'ROI Achievement: >250%',
            'Operational Cost Reduction: 35%'
        ]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        for category in list(success_metrics.keys())[:2]:
            st.markdown(f"**{category}:**")
            for metric in success_metrics[category]:
                st.markdown(f"• {metric}")
            st.markdown("")
    
    with col2:
        for category in list(success_metrics.keys())[2:]:
            st.markdown(f"**{category}:**")
            for metric in success_metrics[category]:
                st.markdown(f"• {metric}")
            st.markdown("")
    
    # Risk mitigation
    st.subheader("Key Risk Mitigation Strategies")
    
    risk_strategies = [
        "**Skills Gap Risk**: Comprehensive training programs and external partnerships",
        "**Technology Risk**: Phased implementation with fallback procedures",
        "**Change Resistance**: Strong change management and communication plan",
        "**Budget Risk**: ROI-focused prioritization and milestone-based funding",
        "**Operational Risk**: Parallel operations during transition periods"
    ]
    
    for strategy in risk_strategies:
        st.markdown(f"• {strategy}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
    <h3>☁️ Cloud Operations Strategic Resource Planning</h3>
    <p><strong>Back Office Cloud Operations & Database Operations</strong> | 5-Year Strategic Plan</p>
    <p><em>Last Updated: August 29, 2025</em></p>
</div>
""", unsafe_allow_html=True)