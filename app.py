# app.py - FIXED Streamlit Dashboard dengan NEW REQUIREMENTS
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys
import tempfile
import io
import time
import traceback

# Import analysis modules
sys.path.append('.')
try:
    from Chat_Analyzer_System import (
        DataPreprocessor, CompleteAnalysisPipeline, Config
    )
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Analysis modules not available: {e}")
    ANALYSIS_AVAILABLE = False

st.set_page_config(
    page_title="Live Chat Analysis Dashboard - NEW REQUIREMENTS",
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #2f0757 100%);
        border-radius: 12px;
        padding: 25px 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        margin: 8px;
        text-align: center;
        border: none;
        color: white;
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.2);
    }
    .metric-card h3 {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.9);
        margin: 0 0 12px 0;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card h1 {
        font-size: 2.2rem;
        color: white;
        margin: 0;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .special-case {
        background-color: #e7f3ff;
        border-left: 4px solid #007bff;
        padding: 10px;
        margin: 5px 0;
        border-radius: 4px;
    }
    .complaint-case {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px;
        margin: 5px 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

def main_interface():
    """Main interface dengan upload dua file"""
    st.markdown('<h1 class="main-header">🤖 Live Chat Analysis - NEW REQUIREMENTS</h1>', unsafe_allow_html=True)
    st.markdown("Upload both conversation data and complaint data for complete analysis")
    
    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'analysis_stats' not in st.session_state:
        st.session_state.analysis_stats = None
    
    # Sidebar untuk upload
    with st.sidebar:
        st.header("📁 Data Upload (REQUIRED BOTH)")
        
        # File 1: Raw Conversation
        st.subheader("1. Raw Conversation Data")
        conversation_file = st.file_uploader(
            "Upload Conversation Excel File", 
            type=['xlsx', 'xls'],
            help="Format: No, Ticket Number, Role, Sender, Message Date, Message"
        )
        
        # File 2: Complaint Data  
        st.subheader("2. Complaint Data")
        complaint_file = st.file_uploader(
            "Upload Complaint Excel File",
            type=['xlsx', 'xls'],
            help="Format: Must contain No.Handphone, Lead Time (Solved), Ticket Number columns"
        )
        
        if conversation_file is not None:
            st.success(f"✅ Conversation file: {conversation_file.name}")
            
        if complaint_file is not None:
            st.success(f"✅ Complaint file: {complaint_file.name}")
        
        st.markdown("---")
        st.header("⚙️ Analysis Settings")
        
        max_tickets = st.slider(
            "Maximum Tickets to Analyze",
            min_value=10,
            max_value=1000,
            value=100,
            help="Limit number of tickets for faster analysis"
        )
        
        st.markdown("---")
        
        # Analysis Button
        if conversation_file is not None and complaint_file is not None:
            if st.button("🚀 START ANALYSIS", type="primary", use_container_width=True):
                with st.spinner("🔄 Starting analysis with new requirements..."):
                    results, stats = run_analysis(conversation_file, complaint_file, max_tickets)
                    
                    if results is not None and stats is not None:
                        st.session_state.analysis_complete = True
                        st.session_state.analysis_results = results
                        st.session_state.analysis_stats = stats
                        st.rerun()
                    else:
                        st.error("❌ Analysis failed. Please check your data format.")
        else:
            st.warning("⚠️ Please upload both files to start analysis")
        
        st.markdown("---")
        st.markdown("### 📖 New Requirements")
        st.info("""
        **Logic Changes:**
        - 1 main question per ticket
        - Support roles: Customer, Operator, Ticket Automation, Blank
        - Complaint matching via phone number
        - Customer leave detection (3 min timeout)
        - Simplified issue classification
        """)

def run_analysis(conversation_file, complaint_file, max_tickets):
    """Run analysis dengan dua file"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Loading files...")
        progress_bar.progress(20)
        
        # 1. LOAD CONVERSATION DATA
        df = pd.read_excel(conversation_file)
        st.success(f"✅ Loaded {len(df)} conversation rows")
        
        # 2. LOAD COMPLAINT DATA
        complaint_df = pd.read_excel(complaint_file)
        st.success(f"✅ Loaded {len(complaint_df)} complaint records")
        
        # 3. CHECK REQUIRED COLUMNS
        required_conv_cols = ['Ticket Number', 'Role', 'Sender', 'Message Date', 'Message']
        missing_conv_cols = [col for col in required_conv_cols if col not in df.columns]
        
        required_comp_cols = ['No.Handphone', 'Lead Time (Solved)']
        missing_comp_cols = [col for col in required_comp_cols if col not in complaint_df.columns]
        
        if missing_conv_cols:
            st.error(f"❌ Missing columns in conversation data: {missing_conv_cols}")
            return None, None
            
        if missing_comp_cols:
            st.error(f"❌ Missing columns in complaint data: {missing_comp_cols}")
            return None, None
        
        status_text.text("🔄 Initializing analysis pipeline...")
        progress_bar.progress(40)
        
        # 4. INITIALIZE AND RUN PIPELINE
        pipeline = CompleteAnalysisPipeline()
        
        # Load complaint data ke pipeline
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_complaint:
            complaint_df.to_excel(tmp_complaint.name, index=False)
            pipeline.load_complaint_data(tmp_complaint.name)
        
        status_text.text("🔍 Analyzing conversations...")
        progress_bar.progress(60)
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        processed_df = preprocessor.clean_data(df)
        
        # Run analysis
        results, stats = pipeline.analyze_all_tickets(processed_df, max_tickets=max_tickets)
        
        status_text.text("💾 Generating results...")
        progress_bar.progress(80)
        
        if not results:
            st.error("❌ No results generated")
            return None, None
            
        successful = [r for r in results if r['status'] == 'success']
        st.success(f"✅ Successfully analyzed {len(successful)} conversations")
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return results, stats
        
    except Exception as e:
        st.error(f"❌ Analysis error: {str(e)}")
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None, None

def display_complete_results():
    """Display analysis results dengan new requirements"""
    
    results = st.session_state.analysis_results
    stats = st.session_state.analysis_stats
    
    if not stats:
        st.error("❌ No analysis statistics available")
        return
    
    st.markdown("---")
    st.markdown('<h1 class="main-header">📊 Analysis Results - NEW REQUIREMENTS</h1>', unsafe_allow_html=True)
    
    # Quick Stats
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Total Tickets</h3>
            <h1>{}</h1>
        </div>
        """.format(stats.get('total_tickets', 0)), unsafe_allow_html=True)

    with col2:
        success_rate = stats.get('success_rate', 0) * 100
        st.markdown("""
        <div class="metric-card">
            <h3>Success Rate</h3>
            <h1>{:.1f}%</h1>
        </div>
        """.format(success_rate), unsafe_allow_html=True)

    with col3:
        if 'overall_lead_times' in stats:
            avg_lead_time = stats['overall_lead_times'].get('first_reply_avg_minutes', 0)
            metric_value = f"{avg_lead_time:.1f} min"
        else:
            metric_value = "N/A"
        
        st.markdown("""
        <div class="metric-card">
            <h3>Avg First Reply</h3>
            <h1>{}</h1>
        </div>
        """.format(metric_value), unsafe_allow_html=True)

    with col4:
        if 'overall_lead_times' in stats:
            avg_lead_time = stats['overall_lead_times'].get('final_reply_avg_minutes', 0)
            metric_value = f"{avg_lead_time:.1f} min"
        else:
            metric_value = "N/A"
        
        st.markdown("""
        <div class="metric-card">
            <h3>Avg Final Reply</h3>
            <h1>{}</h1>
        </div>
        """.format(metric_value), unsafe_allow_html=True)
        
    with col5:
        if 'issue_type_distribution' in stats:
            complaint_count = stats['issue_type_distribution'].get('complaint', 0)
            metric_value = complaint_count
        else:
            metric_value = "N/A"
        
        st.markdown("""
        <div class="metric-card">
            <h3>Complaints</h3>
            <h1>{}</h1>
        </div>
        """.format(metric_value), unsafe_allow_html=True)

    # SPECIAL CASES SUMMARY
    if 'reply_effectiveness' in stats:
        eff = stats['reply_effectiveness']
        if eff.get('customer_leave_cases', 0) > 0:
            st.markdown("---")
            st.markdown("## 🚨 Special Cases Summary")
            
            st.markdown('<div class="special-case">', unsafe_allow_html=True)
            st.metric("Customer Leave Cases", eff.get('customer_leave_cases', 0))
            st.caption("Conversations where customer left without response (3+ minutes no reply)")
            st.markdown('</div>', unsafe_allow_html=True)

    # COMPLAINT CASES SUMMARY
    successful = [r for r in results if r['status'] == 'success']
    complaint_cases = [r for r in successful if r.get('is_complaint')]
    
    if complaint_cases:
        st.markdown("---")
        st.markdown("## 📋 Complaint Cases Summary")
        
        st.markdown('<div class="complaint-case">', unsafe_allow_html=True)
        st.metric("Complaint Cases Matched", len(complaint_cases))
        st.caption("Cases matched between conversation and complaint data via phone number")
        st.markdown('</div>', unsafe_allow_html=True)

    # TABS
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "🎯 Main Issues", "⏱️ Lead Times", "📋 Complaint Cases", "🔍 All Data"
    ])
    
    with tab1:
        display_overview_tab(results, stats)
    
    with tab2:
        display_main_issues_tab(results)
    
    with tab3:
        display_lead_time_tab(results, stats)
    
    with tab4:
        display_complaint_cases_tab(results)
    
    with tab5:
        display_raw_data_tab(results)

    # NEW ANALYSIS BUTTON
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Analyze New Data", type="secondary", use_container_width=True):
            # Reset session state
            for key in ['analysis_complete', 'analysis_results', 'analysis_stats']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

def display_overview_tab(results, stats):
    """Display overview tab"""
    st.markdown("## 📈 Analysis Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Issue Type Distribution
        if 'issue_type_distribution' in stats:
            issue_types = list(stats['issue_type_distribution'].keys())
            counts = list(stats['issue_type_distribution'].values())
            
            fig_issues = px.pie(
                values=counts, names=issue_types,
                title='Issue Type Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_issues, use_container_width=True)
        else:
            st.info("No issue type distribution data available")
    
    with col2:
        # Performance Distribution
        if 'performance_distribution' in stats:
            performances = list(stats['performance_distribution'].keys())
            counts = list(stats['performance_distribution'].values())
            
            fig_perf = px.bar(
                x=performances, y=counts,
                title='Performance Rating Distribution',
                labels={'x': 'Performance Rating', 'y': 'Count'},
                color=performances,
                color_discrete_map={
                    'good': '#28a745',
                    'fair': '#ffc107',
                }
            )
            st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.info("No performance distribution data available")
    
    # Summary Statistics
    st.markdown("### 📊 Summary Statistics")
    
    # Lead Time Summary
    if 'overall_lead_times' in stats:
        lt_stats = stats['overall_lead_times']
        st.markdown("#### ⏱️ Overall Lead Time Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("First Reply Avg", f"{lt_stats['first_reply_avg_minutes']:.1f} min")
        with col2:
            st.metric("Final Reply Avg", f"{lt_stats['final_reply_avg_minutes']:.1f} min")
        with col3:
            st.metric("First Reply Samples", lt_stats['first_reply_samples'])
        with col4:
            st.metric("Final Reply Samples", lt_stats['final_reply_samples'])
    
    # Reply Effectiveness
    if 'reply_effectiveness' in stats:
        eff = stats['reply_effectiveness']
        st.markdown("#### 💬 Reply Effectiveness")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("First Reply Found", f"{eff['first_reply_found_rate']*100:.1f}%")
        with col2:
            st.metric("Final Reply Found", f"{eff['final_reply_found_rate']*100:.1f}%")
        with col3:
            st.metric("Customer Leave Cases", eff['customer_leave_cases'])

def display_main_issues_tab(results):
    """Display main issues dengan new logic"""
    st.markdown("## 🎯 Main Issues Analysis (1 Question per Ticket)")
    
    successful = [r for r in results if r['status'] == 'success']
    
    if successful:
        # Summary cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            first_reply_found = sum(1 for r in successful if r['first_reply_found'])
            st.metric("First Reply Found", f"{first_reply_found}/{len(successful)}")
        
        with col2:
            final_reply_found = sum(1 for r in successful if r['final_reply_found'])
            st.metric("Final Reply Found", f"{final_reply_found}/{len(successful)}")
        
        with col3:
            requirement_met = sum(1 for r in successful if r['requirement_compliant'])
            st.metric("Requirements Met", f"{requirement_met}/{len(successful)}")
        
        with col4:
            complaint_cases = sum(1 for r in successful if r.get('is_complaint'))
            st.metric("Complaint Cases", f"{complaint_cases}/{len(successful)}")

        # Main issues table
        st.markdown("### 📋 All Main Issues")
        display_data = []
        for result in successful:
            special_notes = []
            if result.get('customer_leave'):
                special_notes.append("🚶 Customer Leave")
            if result.get('is_complaint'):
                special_notes.append("📋 Complaint")
            
            display_data.append({
                'Ticket ID': result['ticket_id'],
                'Main Question': result['main_question'][:80] + '...' if len(result['main_question']) > 80 else result['main_question'],
                'Issue Type': result['final_issue_type'].upper(),
                'First Reply': '✅' if result['first_reply_found'] else '❌',
                'Final Reply': '✅' if result['final_reply_found'] else '❌',
                'First Reply LT (min)': result.get('first_reply_lead_time_minutes', 'N/A'),
                'Final Reply LT (min)': result.get('final_reply_lead_time_minutes', 'N/A'),
                'Performance': result['performance_rating'].upper(),
                'Special Notes': ', '.join(special_notes) if special_notes else '-'
            })
        
        df_display = pd.DataFrame(display_data)
        st.dataframe(df_display, use_container_width=True)
        
        # Detailed view for selected ticket
        st.markdown("### 🔍 Detailed Conversation View")
        ticket_options = [f"{r['ticket_id']} - {r['main_question'][:50]}..." for r in successful]
        selected_ticket = st.selectbox("Select ticket for details:", ticket_options, key="ticket_selector")
        
        if selected_ticket:
            ticket_id = selected_ticket.split(' - ')[0]
            selected_result = next((r for r in successful if r['ticket_id'] == ticket_id), None)
            
            if selected_result:
                display_ticket_details(selected_result)
    else:
        st.info("No successful analyses to display")

def display_ticket_details(result):
    """Display detailed information for a single ticket"""
    
    # Main Question
    st.markdown("### 📝 Main Question")
    st.markdown(f'<div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff;"><strong>Question:</strong> {result["main_question"]}</div>', unsafe_allow_html=True)
    st.caption(f"Detected as: {result['final_issue_type'].upper()}")
    
    # SPECIAL CASES SECTION
    special_notes = []
    if result.get('customer_leave'):
        special_notes.append("🚶 **Customer Leave**: Customer left conversation without response (3+ minutes)")
    if result.get('is_complaint'):
        special_notes.append("📋 **Complaint Case**: Matched with complaint data via phone number")
    
    if special_notes:
        st.markdown("### 🚨 Special Conditions")
        for note in special_notes:
            st.markdown(f'<div class="special-case">{note}</div>', unsafe_allow_html=True)
    
    # First Reply Section
    st.markdown("#### 🔄 First Reply Analysis")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if result['first_reply_found']:
            first_reply_msg = result.get('first_reply_message', 'No message content available')
            st.markdown(f'<div style="background-color: #e8f5e8; padding: 15px; border-radius: 5px; border-left: 4px solid #28a745;"><strong>First Reply:</strong> {first_reply_msg}</div>', unsafe_allow_html=True)
        else:
            if result['final_issue_type'] in ['serious', 'complaint']:
                st.error("❌ No first reply found - REQUIRED for serious/complaint issues")
            else:
                st.info("ℹ️ No first reply found - Not required for normal issues")
    
    with col2:
        if result['first_reply_found']:
            st.metric("Lead Time", f"{result.get('first_reply_lead_time_minutes', 'N/A')} min")
            st.metric("Time Format", result.get('first_reply_lead_time_hhmmss', 'N/A'))
        else:
            st.metric("Status", "Not Found")
    
    # Final Reply Section  
    st.markdown("#### ✅ Final Reply Analysis")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if result['final_reply_found']:
            final_reply_msg = result.get('final_reply_message', 'No message content available')
            
            # Special handling untuk complaint
            if result.get('is_complaint') and final_reply_msg == 'COMPLAINT_RESOLVED':
                st.markdown(f'<div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107;"><strong>Final Reply (Complaint Resolution):</strong> Resolved in {result.get("final_lead_time_days", "N/A")} days (from complaint data)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="background-color: #e8f5e8; padding: 15px; border-radius: 5px; border-left: 4px solid #28a745;"><strong>Final Reply:</strong> {final_reply_msg}</div>', unsafe_allow_html=True)
        else:
            if result['final_issue_type'] == 'normal' and not result.get('customer_leave'):
                st.error("❌ No final reply found - REQUIRED for normal issues")
            else:
                st.info("ℹ️ No final reply found - May be handled in follow-up")
    
    with col2:
        if result['final_reply_found']:
            if result.get('is_complaint') and result.get('final_lead_time_days'):
                st.metric("Lead Time", f"{result['final_lead_time_days']} days")
            else:
                st.metric("Lead Time", f"{result.get('final_reply_lead_time_minutes', 'N/A')} min")
                st.metric("Time Format", result.get('final_reply_lead_time_hhmmss', 'N/A'))
        else:
            st.metric("Status", "Not Found")
    
    # Performance Metrics
    st.markdown("#### 📊 Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        performance_color = "#28a745" if result['performance_rating'] == 'good' else "#ffc107"
        st.markdown(f"""
        <div style="background-color: {performance_color}; color: white; padding: 15px; border-radius: 10px; text-align: center;">
            <h3 style="margin: 0; font-size: 1.2rem;">Performance Rating</h3>
            <h1 style="margin: 10px 0; font-size: 2.5rem;">{result['performance_rating'].upper()}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        quality_color = "#28a745" if result['quality_score'] >= 3 else "#ffc107"
        st.markdown(f"""
        <div style="background-color: {quality_color}; color: white; padding: 15px; border-radius: 10px; text-align: center;">
            <h3 style="margin: 0; font-size: 1.2rem;">Quality Score</h3>
            <h1 style="margin: 10px 0; font-size: 2.5rem;">{result['quality_score']}/4</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.metric("Requirement Compliant", "✅" if result['requirement_compliant'] else "❌")
    
    with col4:
        st.metric("Total Messages", result['total_messages'])
    
    # Complaint Data (jika ada)
    if result.get('complaint_data'):
        st.markdown("#### 📋 Complaint Data")
        complaint_data = result['complaint_data']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Complaint Phone", complaint_data.get('complaint_phone', 'N/A'))
        with col2:
            st.metric("Lead Time (Days)", complaint_data.get('lead_time_days', 'N/A'))
        with col3:
            st.metric("Complaint Ticket", complaint_data.get('ticket_number', 'N/A'))

def display_lead_time_tab(results, stats):
    """Display lead time analysis"""
    st.markdown("## ⏱️ Lead Time Analysis")
    
    successful = [r for r in results if r['status'] == 'success']
    
    # Overall Lead Time Summary
    if 'overall_lead_times' in stats:
        overall_lt = stats['overall_lead_times']
        st.markdown("### 📊 Overall Lead Time Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("First Reply Average", f"{overall_lt['first_reply_avg_minutes']:.1f} min")
        with col2:
            st.metric("Final Reply Average", f"{overall_lt['final_reply_avg_minutes']:.1f} min")
        with col3:
            st.metric("First Reply Samples", overall_lt['first_reply_samples'])
        with col4:
            st.metric("Final Reply Samples", overall_lt['final_reply_samples'])
    
    # Lead Time by Issue Type
    if successful:
        st.markdown("### 📈 Lead Time by Issue Type")
        
        issue_data = {}
        for result in successful:
            issue_type = result['final_issue_type']
            first_lt = result.get('first_reply_lead_time_minutes')
            final_lt = result.get('final_reply_lead_time_minutes')
            
            if issue_type not in issue_data:
                issue_data[issue_type] = {'first_times': [], 'final_times': []}
            
            if first_lt is not None:
                issue_data[issue_type]['first_times'].append(first_lt)
            if final_lt is not None:
                issue_data[issue_type]['final_times'].append(final_lt)
        
        # Create comparison chart
        comparison_data = []
        for issue_type, times in issue_data.items():
            if times['first_times']:
                comparison_data.append({
                    'Issue Type': issue_type.upper(),
                    'Lead Time Type': 'First Reply',
                    'Average Lead Time (min)': np.mean(times['first_times']),
                    'Samples': len(times['first_times'])
                })
            if times['final_times']:
                comparison_data.append({
                    'Issue Type': issue_type.upper(),
                    'Lead Time Type': 'Final Reply', 
                    'Average Lead Time (min)': np.mean(times['final_times']),
                    'Samples': len(times['final_times'])
                })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            
            fig = px.bar(
                df_comparison, 
                x='Issue Type', 
                y='Average Lead Time (min)', 
                color='Lead Time Type',
                title='Average Lead Time by Issue Type and Reply Type',
                barmode='group',
                color_discrete_map={
                    'First Reply': '#2E86AB',
                    'Final Reply': '#A23B72'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Lead Time Distribution
    st.markdown("### 📊 Detailed Lead Time Distribution")
    
    successful_with_lead_times = [r for r in successful if r.get('final_reply_lead_time_minutes') is not None]
    
    if successful_with_lead_times:
        lead_times = [r['final_reply_lead_time_minutes'] for r in successful_with_lead_times]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig_hist = px.histogram(
                x=lead_times,
                title='Final Reply Lead Time Distribution',
                labels={'x': 'Lead Time (minutes)', 'y': 'Frequency'},
                nbins=20,
                color_discrete_sequence=['#2E86AB']
            )
            fig_hist.add_vline(
                x=np.mean(lead_times), 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Mean: {np.mean(lead_times):.1f} min"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Statistics
            st.markdown("#### 📈 Lead Time Statistics")
            st.metric("Mean", f"{np.mean(lead_times):.1f} min")
            st.metric("Median", f"{np.median(lead_times):.1f} min") 
            st.metric("Std Dev", f"{np.std(lead_times):.1f} min")
            st.metric("Range", f"{np.min(lead_times):.1f} - {np.max(lead_times):.1f} min")
    else:
        st.info("No lead time data available for detailed analysis")

def display_complaint_cases_tab(results):
    """Display complaint cases analysis"""
    st.markdown("## 📋 Complaint Cases Analysis")
    
    successful = [r for r in results if r['status'] == 'success']
    complaint_cases = [r for r in successful if r.get('is_complaint')]
    
    if complaint_cases:
        st.success(f"✅ Found {len(complaint_cases)} complaint cases matched via phone number")
        
        # Complaint cases table
        complaint_data = []
        for result in complaint_cases:
            complaint_info = result.get('complaint_data', {})
            
            complaint_data.append({
                'Ticket ID': result['ticket_id'],
                'Main Question': result['main_question'][:80] + '...',
                'Matched Phone': complaint_info.get('complaint_phone', 'N/A'),
                'Complaint Ticket': complaint_info.get('ticket_number', 'N/A'),
                'First Reply Found': '✅' if result['first_reply_found'] else '❌',
                'Final Resolution Time': f"{result.get('final_lead_time_days', 'N/A')} days",
                'First Reply LT (min)': result.get('first_reply_lead_time_minutes', 'N/A')
            })
        
        df_complaints = pd.DataFrame(complaint_data)
        st.dataframe(df_complaints, use_container_width=True)
        
        # Complaint insights
        st.markdown("### 💡 Complaint Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            first_reply_rate = sum(1 for r in complaint_cases if r['first_reply_found']) / len(complaint_cases) * 100
            st.metric("First Reply Rate", f"{first_reply_rate:.1f}%")
        
        with col2:
            avg_resolution_days = np.mean([r.get('final_lead_time_days', 0) for r in complaint_cases if r.get('final_lead_time_days')])
            st.metric("Avg Resolution Time", f"{avg_resolution_days:.1f} days")
        
        with col3:
            requirement_met = sum(1 for r in complaint_cases if r['requirement_compliant'])
            st.metric("Requirements Met", f"{requirement_met}/{len(complaint_cases)}")
        
    else:
        st.info("No complaint cases found. Make sure phone numbers in conversation data match complaint data.")

def display_raw_data_tab(results):
    """Display raw data"""
    st.markdown("## 📋 All Analyzed Data")
    
    successful = [r for r in results if r['status'] == 'success']
    
    if successful:
        st.info(f"Showing {len(successful)} successful analyses with new requirement logic.")
        
        # Tampilkan data lengkap
        raw_data = []
        for result in successful:
            special_notes = []
            if result.get('customer_leave'):
                special_notes.append("Customer Leave")
            if result.get('is_complaint'):
                special_notes.append("Complaint")
            
            raw_data.append({
                'Ticket ID': result['ticket_id'],
                'Main Question': result['main_question'],
                'Issue Type': result['final_issue_type'],
                'First Reply Found': result['first_reply_found'],
                'First Reply Message': result.get('first_reply_message', '')[:100] + '...' if result.get('first_reply_message') else 'Not found',
                'First Reply LT (min)': result.get('first_reply_lead_time_minutes'),
                'Final Reply Found': result['final_reply_found'],
                'Final Reply Message': result.get('final_reply_message', '')[:100] + '...' if result.get('final_reply_message') else 'Not found',
                'Final Reply LT (min)': result.get('final_reply_lead_time_minutes'),
                'Final Reply LT (Days)': result.get('final_lead_time_days'),
                'Performance': result['performance_rating'],
                'Quality Score': result['quality_score'],
                'Requirement Compliant': result['requirement_compliant'],
                'Special Notes': ', '.join(special_notes) if special_notes else 'None'
            })
        
        df_raw = pd.DataFrame(raw_data)
        st.dataframe(df_raw, use_container_width=True)
        
    else:
        st.info("No successful analyses to display")

# Main execution
if __name__ == "__main__":
    if not ANALYSIS_AVAILABLE:
        st.error("""
        ❌ Analysis modules not available!
        
        Please ensure `Chat_Analyzer_System.py` exists with all required classes.
        """)
        st.stop()
    
    # Check if analysis is complete
    if st.session_state.get('analysis_complete', False):
        display_complete_results()
    else:
        main_interface()


