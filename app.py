# app.py - Streamlit Dashboard dengan Logic Baru
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys
import tempfile
import traceback

# Import analysis modules
sys.path.append('.')
try:
    from chat_analyzer import (
        DataPreprocessor, CompleteAnalysisPipeline, 
        ResultsExporter, Config
    )
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Analysis modules not available: {e}")
    ANALYSIS_AVAILABLE = False

st.set_page_config(
    page_title="Live Chat Analysis Dashboard - Enhanced",
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
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .message-box {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
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
    st.markdown('<h1 class="main-header">🤖 Enhanced Live Chat Analysis</h1>', unsafe_allow_html=True)
    st.markdown("Upload both conversation data and complaint data for comprehensive analysis")
    
    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'analysis_stats' not in st.session_state:
        st.session_state.analysis_stats = None
    if 'excel_file_path' not in st.session_state:
        st.session_state.excel_file_path = None
    
    # Sidebar untuk upload
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # File 1: Raw Conversation Data
        st.subheader("1. Raw Conversation Data")
        uploaded_raw_file = st.file_uploader(
            "Upload Raw Conversation Excel", 
            type=['xlsx', 'xls'],
            help="Format: No, Ticket Number, Role, Sender, Message Date, Message",
            key="raw_uploader"
        )
        
        # File 2: Complaint Data
        st.subheader("2. Complaint Data")
        uploaded_complaint_file = st.file_uploader(
            "Upload Complaint Data Excel", 
            type=['xlsx', 'xls'],
            help="Format dengan kolom No.Handphone dan Lead Time (Solved)",
            key="complaint_uploader"
        )
        
        if uploaded_raw_file is not None:
            st.success(f"✅ Raw file: {uploaded_raw_file.name}")
            
        if uploaded_complaint_file is not None:
            st.success(f"✅ Complaint file: {uploaded_complaint_file.name}")
        
        # Preview data
        if st.checkbox("Preview uploaded data"):
            if uploaded_raw_file is not None:
                try:
                    df_preview = pd.read_excel(uploaded_raw_file)
                    st.write(f"Raw data shape: {df_preview.shape}")
                    st.dataframe(df_preview.head(5))
                except Exception as e:
                    st.error(f"Error previewing raw data: {e}")
            
            if uploaded_complaint_file is not None:
                try:
                    df_complaint_preview = pd.read_excel(uploaded_complaint_file)
                    st.write(f"Complaint data shape: {df_complaint_preview.shape}")
                    st.dataframe(df_complaint_preview.head(5))
                except Exception as e:
                    st.error(f"Error previewing complaint data: {e}")
        
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
        if uploaded_raw_file is not None and uploaded_complaint_file is not None:
            if st.button("🚀 START ENHANCED ANALYSIS", type="primary", use_container_width=True):
                with st.spinner("🔄 Starting enhanced analysis..."):
                    results, stats, excel_path = run_enhanced_analysis(
                        uploaded_raw_file, uploaded_complaint_file, max_tickets
                    )
                    
                    if results is not None and stats is not None:
                        st.session_state.analysis_complete = True
                        st.session_state.analysis_results = results
                        st.session_state.analysis_stats = stats
                        st.session_state.excel_file_path = excel_path
                        st.rerun()
                    else:
                        st.error("❌ Analysis failed. Please check your data format.")
        else:
            st.warning("⚠️ Please upload both files to start analysis")
        
        st.markdown("---")
        st.markdown("### 📖 How to Use")
        st.info("""
        **Enhanced Analysis Features:**
        1. **New Role Handling**: Ticket Automation & Blank roles
        2. **Complaint Detection**: Matches phones between files
        3. **Smart Issue Typing**: 
           - Normal: Direct solutions
           - Serious: Ticket reopened cases  
           - Complaint: From complaint system
        4. **Customer Leave Detection**: Based on automation messages
        
        **Required Files:**
        - Raw Conversation Data
        - Complaint Data (with No.Handphone)
        """)
    
    # Main content area
    if uploaded_raw_file is not None and uploaded_complaint_file is not None:
        st.markdown("---")
        st.markdown("## 🚀 Ready for Enhanced Analysis")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🎯 Start Enhanced Analysis", type="primary", use_container_width=True):
                with st.spinner("🔄 Starting enhanced analysis..."):
                    results, stats, excel_path = run_enhanced_analysis(
                        uploaded_raw_file, uploaded_complaint_file, max_tickets
                    )
                    
                    if results is not None and stats is not None:
                        st.session_state.analysis_complete = True
                        st.session_state.analysis_results = results
                        st.session_state.analysis_stats = stats
                        st.session_state.excel_file_path = excel_path
                        st.rerun()
                    else:
                        st.error("❌ Analysis failed. Please check your data format.")
    
    else:
        # Show upload reminder
        st.markdown("---")
        st.markdown("## 📤 Upload Required Files")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **Raw Conversation Data**
            - Format: Excel file
            - Required columns: 
              - Ticket Number
              - Role  
              - Sender
              - Message Date
              - Message
            """)
        
        with col2:
            st.info("""
            **Complaint Data** 
            - Format: Excel file
            - Required columns:
              - No.Handphone
              - Lead Time (Solved)
            - For complaint ticket matching
            """)

def run_enhanced_analysis(uploaded_raw_file, uploaded_complaint_file, max_tickets):
    """Run enhanced analysis dengan kedua file"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Load Data
        status_text.text("📊 Loading and processing data...")
        progress_bar.progress(20)
        
        # Load raw data
        raw_df = pd.read_excel(uploaded_raw_file)
        st.success(f"✅ Loaded {len(raw_df)} rows from raw conversation data")
        
        # Load complaint data
        complaint_df = pd.read_excel(uploaded_complaint_file)
        st.success(f"✅ Loaded {len(complaint_df)} rows from complaint data")
        
        # Step 2: Initialize Pipeline dengan Complaint Data
        status_text.text("🔧 Initializing enhanced analysis pipeline...")
        progress_bar.progress(40)
        
        # Simpan complaint file temporary
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_complaint:
            complaint_df.to_excel(tmp_complaint.name, index=False)
            pipeline = CompleteAnalysisPipeline(complaint_data_path=tmp_complaint.name)
        
        # Step 3: Preprocess Data
        status_text.text("🔄 Preprocessing conversation data...")
        progress_bar.progress(60)
        
        preprocessor = DataPreprocessor()
        processed_df = preprocessor.clean_data(raw_df)
        
        # Step 4: Run Analysis
        status_text.text("🔍 Analyzing conversations with enhanced logic...")
        progress_bar.progress(80)
        
        results, stats = pipeline.analyze_all_tickets(processed_df, max_tickets=max_tickets)
        
        if not results:
            st.error("❌ No results generated from analysis")
            return None, None, None
            
        successful = [r for r in results if r['status'] == 'success']
        st.success(f"✅ Successfully analyzed {len(successful)} conversations")
        
        # Step 5: Export Results
        status_text.text("💾 Generating comprehensive report...")
        progress_bar.progress(95)
        
        exporter = ResultsExporter()
        excel_path = exporter.export_comprehensive_results(results, stats)
        
        progress_bar.progress(100)
        status_text.text("✅ Enhanced analysis complete!")
        
        return results, stats, excel_path
        
    except Exception as e:
        st.error(f"❌ Enhanced analysis error: {str(e)}")
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None, None, None

def display_enhanced_results():
    """Display enhanced analysis results"""
    results = st.session_state.analysis_results
    stats = st.session_state.analysis_stats
    
    if not stats:
        st.error("❌ No analysis statistics available")
        return
    
    st.markdown("---")
    st.markdown('<h1 class="main-header">📊 Enhanced Analysis Results</h1>', unsafe_allow_html=True)
    
    # Quick Stats dengan complaint info
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Tickets</h3>
            <h1>{stats.get('total_tickets', 0)}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        success_rate = stats.get('success_rate', 0) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>Success Rate</h3>
            <h1>{success_rate:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        if 'lead_time_stats' in stats:
            avg_lead_time = stats['lead_time_stats'].get('first_reply_avg_minutes', 0)
            metric_value = f"{avg_lead_time:.1f} min"
        else:
            metric_value = "N/A"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg First Reply</h3>
            <h1>{metric_value}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        if 'lead_time_stats' in stats:
            avg_lead_time = stats['lead_time_stats'].get('final_reply_avg_minutes', 0)
            metric_value = f"{avg_lead_time:.1f} min"
        else:
            metric_value = "N/A"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Final Reply</h3>
            <h1>{metric_value}</h1>
        </div>
        """, unsafe_allow_html=True)
        
    with col5:
        if 'issue_type_distribution' in stats:
            complaint_count = stats['issue_type_distribution'].get('complaint', 0)
            metric_value = complaint_count
        else:
            metric_value = "N/A"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Complaint Cases</h3>
            <h1>{metric_value}</h1>
        </div>
        """, unsafe_allow_html=True)

    # SPECIAL CASES SUMMARY - Enhanced
    if 'reply_effectiveness' in stats:
        eff = stats['reply_effectiveness']
        complaint_cases = stats.get('issue_type_distribution', {}).get('complaint', 0)
        
        if eff.get('customer_leave_cases', 0) > 0 or complaint_cases > 0:
            st.markdown("---")
            st.markdown("## 🚨 Special Cases Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="special-case">', unsafe_allow_html=True)
                st.metric("Customer Leave Cases", eff.get('customer_leave_cases', 0))
                st.caption("Detected by Ticket Automation messages")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="complaint-case">', unsafe_allow_html=True)
                st.metric("Complaint Cases", complaint_cases)
                st.caption("Matched from complaint system")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                serious_cases = stats.get('issue_type_distribution', {}).get('serious', 0)
                st.markdown('<div class="special-case">', unsafe_allow_html=True)
                st.metric("Serious Cases", serious_cases)
                st.caption("With ticket reopened")
                st.markdown('</div>', unsafe_allow_html=True)

    # DOWNLOAD SECTION
    st.markdown("---")
    st.markdown("## 💾 Download Enhanced Analysis Report")
    
    if st.session_state.get('excel_file_path') and os.path.exists(st.session_state.excel_file_path):
        with open(st.session_state.excel_file_path, "rb") as f:
            excel_data = f.read()
        
        st.download_button(
            label="📥 DOWNLOAD ENHANCED EXCEL REPORT",
            data=excel_data,
            file_name=f"enhanced_chat_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True
        )
        st.success("✅ Enhanced report contains: Complaint matching, new issue typing, lead times in days/hours")
    else:
        st.error("❌ Excel file not available for download")

    # TABS - Enhanced
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview", "🎯 Issue Types", "⏱️ Lead Times", "📊 Performance", "🚨 Special Cases", "📋 All Data"
    ])
    
    with tab1:
        display_enhanced_overview_tab(results, stats)
    
    with tab2:
        display_issue_types_tab(results, stats)
    
    with tab3:
        display_enhanced_lead_time_tab(results, stats)
    
    with tab4:
        display_performance_tab(results, stats)
    
    with tab5:
        display_enhanced_special_cases_tab(results, stats)
    
    with tab6:
        display_raw_data_tab(results)

    # NEW ANALYSIS BUTTON
    st.markdown("---")
    if st.button("🔄 Analyze New Data", type="secondary", use_container_width=True):
        # Reset session state
        for key in ['analysis_complete', 'analysis_results', 'analysis_stats', 'excel_file_path']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

def display_enhanced_overview_tab(results, stats):
    """Display enhanced overview tab"""
    st.markdown("## 📈 Enhanced Analysis Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Issue Type Distribution
        if 'issue_type_distribution' in stats:
            issue_types = list(stats['issue_type_distribution'].keys())
            counts = list(stats['issue_type_distribution'].values())
            
            colors = ['#2E86AB', '#A23B72', '#F18F01']  # normal, serious, complaint
            color_map = {}
            for i, issue_type in enumerate(issue_types):
                if i < len(colors):
                    color_map[issue_type] = colors[i]
                else:
                    color_map[issue_type] = '#999999'
            
            fig_issues = px.pie(
                values=counts, names=issue_types,
                title='Issue Type Distribution (Enhanced)',
                color=issue_types,
                color_discrete_map=color_map
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
                    'excellent': '#2E86AB',
                    'good': '#A23B72', 
                    'fair': '#F18F01',
                    'poor': '#C73E1D'
                }
            )
            st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.info("No performance distribution data available")
    
    # Enhanced Summary Statistics
    st.markdown("### 📊 Enhanced Summary Statistics")
    
    # Lead Time Summary
    if 'lead_time_stats' in stats:
        lt_stats = stats['lead_time_stats']
        st.markdown("#### ⏱️ Lead Time Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("First Reply Avg", f"{lt_stats['first_reply_avg_minutes']:.1f} min")
        with col2:
            st.metric("Final Reply Avg", f"{lt_stats['final_reply_avg_minutes']:.1f} min")
        with col3:
            st.metric("First Reply Samples", lt_stats['first_reply_samples'])
        with col4:
            st.metric("Final Reply Samples", lt_stats['final_reply_samples'])
    
    # Issue Type Breakdown
    if 'issue_type_distribution' in stats:
        st.markdown("#### 🎯 Issue Type Breakdown")
        issue_data = []
        for issue_type, count in stats['issue_type_distribution'].items():
            percentage = (count / stats['successful_analysis']) * 100
            issue_data.append({
                'Issue Type': issue_type.upper(),
                'Count': count,
                'Percentage': f"{percentage:.1f}%"
            })
        
        df_issues = pd.DataFrame(issue_data)
        st.dataframe(df_issues, use_container_width=True)

def display_issue_types_tab(results, stats):
    """Display issue types analysis"""
    st.markdown("## 🎯 Issue Types Analysis")
    
    successful = [r for r in results if r['status'] == 'success']
    
    if successful:
        # Summary by issue type
        st.markdown("### 📊 Summary by Issue Type")
        
        issue_summary = {}
        for result in successful:
            issue_type = result['final_issue_type']
            if issue_type not in issue_summary:
                issue_summary[issue_type] = {
                    'count': 0,
                    'avg_quality': [],
                    'first_reply_found': 0,
                    'final_reply_found': 0
                }
            
            issue_summary[issue_type]['count'] += 1
            issue_summary[issue_type]['avg_quality'].append(result['quality_score'])
            if result['first_reply_found']:
                issue_summary[issue_type]['first_reply_found'] += 1
            if result['final_reply_found']:
                issue_summary[issue_type]['final_reply_found'] += 1
        
        # Display summary
        summary_data = []
        for issue_type, data in issue_summary.items():
            avg_quality = np.mean(data['avg_quality'])
            first_reply_rate = (data['first_reply_found'] / data['count']) * 100
            final_reply_rate = (data['final_reply_found'] / data['count']) * 100
            
            summary_data.append({
                'Issue Type': issue_type.upper(),
                'Count': data['count'],
                'Avg Quality Score': f"{avg_quality:.1f}",
                'First Reply Rate': f"{first_reply_rate:.1f}%",
                'Final Reply Rate': f"{final_reply_rate:.1f}%"
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
        
        # Detailed view
        st.markdown("### 📋 Detailed Issues View")
        
        display_data = []
        for result in successful:
            special_notes = []
            if result.get('customer_leave'):
                special_notes.append("🚶 Customer Leave")
            if result['final_issue_type'] == 'complaint':
                special_notes.append("📋 From Complaint System")
            
            display_data.append({
                'Ticket ID': result['ticket_id'],
                'Issue Type': result['final_issue_type'].upper(),
                'Main Question': result['main_question'][:80] + '...',
                'First Reply': '✅' if result['first_reply_found'] else '❌',
                'Final Reply': '✅' if result['final_reply_found'] else '❌',
                'First LT (min)': result.get('first_reply_lead_time_minutes', 'N/A'),
                'Final LT': self._format_lead_time(result),
                'Performance': result['performance_rating'].upper(),
                'Special Notes': ', '.join(special_notes) if special_notes else '-'
            })
        
        df_display = pd.DataFrame(display_data)
        st.dataframe(df_display, use_container_width=True)
        
        # Detailed view for selected ticket
        st.markdown("### 🔍 Detailed Ticket View")
        ticket_options = [f"{r['ticket_id']} - {r['final_issue_type'].upper()} - {r['main_question'][:50]}..." for r in successful]
        selected_ticket = st.selectbox("Select ticket for details:", ticket_options, key="ticket_selector")
        
        if selected_ticket:
            ticket_id = selected_ticket.split(' - ')[0]
            selected_result = next((r for r in successful if r['ticket_id'] == ticket_id), None)
            
            if selected_result:
                display_enhanced_ticket_details(selected_result)
    else:
        st.info("No successful analyses to display")

def _format_lead_time(self, result):
    """Format lead time berdasarkan jenis issue"""
    if result['final_issue_type'] == 'complaint' and result.get('final_reply_lead_time_days'):
        return f"{result['final_reply_lead_time_days']} days"
    elif result.get('final_reply_lead_time_minutes'):
        return f"{result['final_reply_lead_time_minutes']:.1f} min"
    else:
        return "N/A"

def display_enhanced_ticket_details(result):
    """Display detailed information for a single ticket dengan enhanced info"""
    
    # Main Question
    st.markdown("### 📝 Main Question")
    st.markdown(f'<div class="message-box"><strong>Question:</strong> {result["main_question"]}</div>', unsafe_allow_html=True)
    
    # Issue Type dengan color coding
    issue_type = result['final_issue_type']
    issue_color = {
        'normal': '#28a745',
        'serious': '#dc3545', 
        'complaint': '#ffc107'
    }.get(issue_type, '#6c757d')
    
    st.markdown(f"""
    <div style="background-color: {issue_color}; color: white; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <strong>Issue Type:</strong> {issue_type.upper()}
    </div>
    """, unsafe_allow_html=True)
    
    # Special Notes
    special_notes = []
    if result.get('customer_leave'):
        special_notes.append("🚶 **Customer Leave**: Detected by Ticket Automation")
    if result['final_issue_type'] == 'complaint':
        special_notes.append("📋 **Complaint Case**: Matched from complaint system")
    if result['final_issue_type'] == 'serious':
        special_notes.append("⚠️ **Serious Case**: Ticket was reopened")
    
    if special_notes:
        st.markdown("### 🚨 Special Conditions")
        for note in special_notes:
            st.markdown(f'<div class="special-case">{note}</div>', unsafe_allow_html=True)
    
    # First Reply Section
    if result['final_issue_type'] in ['serious', 'complaint']:
        st.markdown("#### 🔄 First Reply Analysis")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if result['first_reply_found']:
                first_reply_msg = result.get('first_reply_message', 'No message content available')
                st.markdown(f'<div class="message-box"><strong>First Reply:</strong> {first_reply_msg}</div>', unsafe_allow_html=True)
            else:
                st.error("❌ No first reply found - REQUIRED for serious/complaint issues")
        
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
            
            if result['final_issue_type'] == 'complaint':
                note = "From complaint system resolution"
            elif result['final_issue_type'] == 'serious':
                note = "After ticket reopened"
            else:
                note = "Direct solution provided"
            
            st.markdown(f'<div class="message-box"><strong>Final Reply ({note}):</strong> {final_reply_msg}</div>', unsafe_allow_html=True)
        else:
            st.error("❌ No final reply found")
    
    with col2:
        if result['final_reply_found']:
            if result['final_issue_type'] == 'complaint':
                lead_time = result.get('final_reply_lead_time_days', 'N/A')
                st.metric("Lead Time", f"{lead_time} days")
            else:
                st.metric("Lead Time", f"{result.get('final_reply_lead_time_minutes', 'N/A')} min")
                st.metric("Time Format", result.get('final_reply_lead_time_hhmmss', 'N/A'))
        else:
            st.metric("Status", "Not Found")
    
    # Performance Metrics
    st.markdown("#### 📊 Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        performance_color = {
            'excellent': '#28a745',
            'good': '#007bff', 
            'fair': '#ffc107',
            'poor': '#dc3545'
        }.get(result['performance_rating'], '#6c757d')
        
        st.markdown(f"""
        <div style="background-color: {performance_color}; color: white; padding: 15px; border-radius: 10px; text-align: center;">
            <h3 style="margin: 0; font-size: 1.2rem;">Performance</h3>
            <h1 style="margin: 10px 0; font-size: 2.5rem;">{result['performance_rating'].upper()}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        quality_color = "#28a745" if result['quality_score'] >= 4 else "#ffc107" if result['quality_score'] >= 2 else "#dc3545"
        st.markdown(f"""
        <div style="background-color: {quality_color}; color: white; padding: 15px; border-radius: 10px; text-align: center;">
            <h3 style="margin: 0; font-size: 1.2rem;">Quality Score</h3>
            <h1 style="margin: 10px 0; font-size: 2.5rem;">{result['quality_score']}/6</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.metric("Total Messages", result['total_messages'])
    
    with col4:
        answer_rate = (result['answered_pairs'] / result['total_qa_pairs']) * 100 if result['total_qa_pairs'] > 0 else 0
        st.metric("Answer Rate", f"{answer_rate:.1f}%")

def display_enhanced_lead_time_tab(results, stats):
    """Display enhanced lead time analysis"""
    st.markdown("## ⏱️ Enhanced Lead Time Analysis")
    
    successful = [r for r in results if r['status'] == 'success']
    
    # Overall Lead Time Summary
    if 'lead_time_stats' in stats:
        lt_stats = stats['lead_time_stats']
        st.markdown("### 📊 Overall Lead Time Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("First Reply Average", f"{lt_stats['first_reply_avg_minutes']:.1f} min")
        with col2:
            st.metric("Final Reply Average", f"{lt_stats['final_reply_avg_minutes']:.1f} min")
        with col3:
            st.metric("First Reply Samples", lt_stats['first_reply_samples'])
        with col4:
            st.metric("Final Reply Samples", lt_stats['final_reply_samples'])
    
    # Lead Time by Issue Type
    st.markdown("### 📈 Lead Time by Issue Type")
    
    lead_time_by_type = {}
    for result in successful:
        issue_type = result['final_issue_type']
        if issue_type not in lead_time_by_type:
            lead_time_by_type[issue_type] = {
                'first_lead_times': [],
                'final_lead_times': [],
                'final_lead_days': []  # Untuk complaint
            }
        
        if result.get('first_reply_lead_time_minutes'):
            lead_time_by_type[issue_type]['first_lead_times'].append(result['first_reply_lead_time_minutes'])
        
        if result['final_issue_type'] == 'complaint' and result.get('final_reply_lead_time_days'):
            lead_time_by_type[issue_type]['final_lead_days'].append(result['final_reply_lead_time_days'])
        elif result.get('final_reply_lead_time_minutes'):
            lead_time_by_type[issue_type]['final_lead_times'].append(result['final_reply_lead_time_minutes'])
    
    # Display lead time by type
    lead_time_data = []
    for issue_type, data in lead_time_by_type.items():
        if data['first_lead_times']:
            first_avg = np.mean(data['first_lead_times'])
        else:
            first_avg = None
        
        if issue_type == 'complaint' and data['final_lead_days']:
            final_avg = np.mean(data['final_lead_days'])
            final_unit = 'days'
        elif data['final_lead_times']:
            final_avg = np.mean(data['final_lead_times'])
            final_unit = 'min'
        else:
            final_avg = None
            final_unit = 'N/A'
        
        lead_time_data.append({
            'Issue Type': issue_type.upper(),
            'First Reply Avg': f"{first_avg:.1f} min" if first_avg else 'N/A',
            'Final Reply Avg': f"{final_avg:.1f} {final_unit}" if final_avg else 'N/A',
            'First Reply Samples': len(data['first_lead_times']),
            'Final Reply Samples': len(data['final_lead_times']) + len(data['final_lead_days'])
        })
    
    if lead_time_data:
        df_lead_time = pd.DataFrame(lead_time_data)
        st.dataframe(df_lead_time, use_container_width=True)
    
    # Lead Time Distribution
    st.markdown("### 📊 Lead Time Distribution")
    
    normal_final_times = [r['final_reply_lead_time_minutes'] for r in successful 
                         if r['final_issue_type'] == 'normal' and r.get('final_reply_lead_time_minutes')]
    serious_final_times = [r['final_reply_lead_time_minutes'] for r in successful 
                          if r['final_issue_type'] == 'serious' and r.get('final_reply_lead_time_minutes')]
    
    if normal_final_times or serious_final_times:
        col1, col2 = st.columns(2)
        
        with col1:
            # Normal issues lead time
            if normal_final_times:
                fig_normal = px.histogram(
                    x=normal_final_times,
                    title='Normal Issues - Final Reply Lead Time',
                    labels={'x': 'Lead Time (minutes)', 'y': 'Frequency'},
                    nbins=20,
                    color_discrete_sequence=['#2E86AB']
                )
                fig_normal.add_vline(
                    x=np.mean(normal_final_times), 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Mean: {np.mean(normal_final_times):.1f} min"
                )
                st.plotly_chart(fig_normal, use_container_width=True)
        
        with col2:
            # Serious issues lead time
            if serious_final_times:
                fig_serious = px.histogram(
                    x=serious_final_times,
                    title='Serious Issues - Final Reply Lead Time',
                    labels={'x': 'Lead Time (minutes)', 'y': 'Frequency'},
                    nbins=20,
                    color_discrete_sequence=['#A23B72']
                )
                fig_serious.add_vline(
                    x=np.mean(serious_final_times), 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Mean: {np.mean(serious_final_times):.1f} min"
                )
                st.plotly_chart(fig_serious, use_container_width=True)
    else:
        st.info("No lead time data available for distribution analysis")

def display_performance_tab(results, stats):
    """Display performance analysis"""
    st.markdown("## 📊 Performance Analysis")
    
    successful = [r for r in results if r['status'] == 'success']
    
    if successful:
        # Performance by Issue Type
        perf_data = []
        for result in successful:
            perf_data.append({
                'issue_type': result['final_issue_type'],
                'performance': result['performance_rating'],
                'quality_score': result['quality_score']
            })
        
        df_perf = pd.DataFrame(perf_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance by Issue Type
            perf_pivot = pd.crosstab(
                df_perf['issue_type'], 
                df_perf['performance']
            ).reset_index()
            
            if len(perf_pivot) > 1:
                fig_stacked = px.bar(
                    perf_pivot, 
                    x='issue_type',
                    y=perf_pivot.columns[1:].tolist(),
                    title='Performance Rating by Issue Type',
                    labels={'value': 'Count', 'issue_type': 'Issue Type'},
                    barmode='stack'
                )
                st.plotly_chart(fig_stacked, use_container_width=True)
        
        with col2:
            # Quality Score Distribution
            fig_quality = px.histogram(
                df_perf, x='quality_score',
                title='Quality Score Distribution',
                labels={'x': 'Quality Score', 'y': 'Count'},
                nbins=6,
                color_discrete_sequence=['#F18F01']
            )
            st.plotly_chart(fig_quality, use_container_width=True)
        
        # Performance metrics table
        st.markdown("### 📈 Detailed Performance Metrics")
        perf_metrics = []
        for result in successful:
            special_notes = []
            if result.get('customer_leave'):
                special_notes.append("Customer Leave")
            if result['final_issue_type'] == 'complaint':
                special_notes.append("Complaint")
            
            perf_metrics.append({
                'Ticket ID': result['ticket_id'],
                'Issue Type': result['final_issue_type'],
                'Performance': result['performance_rating'].upper(),
                'Quality Score': result['quality_score'],
                'First Reply LT': result.get('first_reply_lead_time_minutes', 'N/A'),
                'Final Reply LT': self._format_lead_time(result),
                'First Reply': '✅' if result['first_reply_found'] else '❌',
                'Final Reply': '✅' if result['final_reply_found'] else '❌',
                'Special Notes': ', '.join(special_notes) if special_notes else '-'
            })
        
        df_perf_metrics = pd.DataFrame(perf_metrics)
        st.dataframe(df_perf_metrics, use_container_width=True)

def display_enhanced_special_cases_tab(results, stats):
    """Display enhanced special cases analysis"""
    st.markdown("## 🚨 Enhanced Special Cases Analysis")
    
    successful = [r for r in results if r['status'] == 'success']
    
    # Extract special cases
    customer_leave_cases = [r for r in successful if r.get('customer_leave')]
    complaint_cases = [r for r in successful if r['final_issue_type'] == 'complaint']
    serious_cases = [r for r in successful if r['final_issue_type'] == 'serious']
    
    # SUMMARY CARDS
    st.markdown("### 📊 Special Cases Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Customer Leave Cases", len(customer_leave_cases))
    
    with col2:
        st.metric("Complaint Cases", len(complaint_cases))
    
    with col3:
        st.metric("Serious Cases", len(serious_cases))
    
    # CUSTOMER LEAVE CASES
    st.markdown("---")
    st.markdown("### 🚶 Customer Leave Cases")
    
    if customer_leave_cases:
        st.info(f"**{len(customer_leave_cases)} conversations** where customer left (detected by Ticket Automation)")
        
        with st.expander("View Customer Leave Details", expanded=True):
            leave_data = []
            for result in customer_leave_cases:
                leave_data.append({
                    'Ticket ID': result['ticket_id'],
                    'Issue Type': result['final_issue_type'].upper(),
                    'Main Question': result['main_question'][:60] + '...',
                    'Final Reply Status': '✅ Found' if result['final_reply_found'] else '❌ Missing',
                    'Performance': result['performance_rating'].upper()
                })
            
            df_leave = pd.DataFrame(leave_data)
            st.dataframe(df_leave, use_container_width=True)
    else:
        st.success("✅ No customer leave cases detected")
    
    # COMPLAINT CASES
    st.markdown("---")
    st.markdown("### 📋 Complaint Cases (From Complaint System)")
    
    if complaint_cases:
        st.warning(f"**{len(complaint_cases)} complaints** matched from complaint system")
        
        with st.expander("View Complaint Details", expanded=True):
            complaint_data = []
            for result in complaint_cases:
                lead_time_days = result.get('final_reply_lead_time_days', 'N/A')
                
                complaint_data.append({
                    'Ticket ID': result['ticket_id'],
                    'Main Question': result['main_question'][:60] + '...',
                    'First Reply Status': '✅ Found' if result['first_reply_found'] else '❌ Missing',
                    'Final Resolution Time': f"{lead_time_days} days",
                    'Performance': result['performance_rating'].upper()
                })
            
            df_complaint = pd.DataFrame(complaint_data)
            st.dataframe(df_complaint, use_container_width=True)
        
        # Complaint analysis
        st.markdown("#### 📈 Complaint Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            avg_lead_time = np.mean([r.get('final_reply_lead_time_days', 0) for r in complaint_cases if r.get('final_reply_lead_time_days')])
            st.metric("Avg Resolution Time", f"{avg_lead_time:.1f} days")
        
        with col2:
            first_reply_rate = sum(1 for r in complaint_cases if r['first_reply_found']) / len(complaint_cases) * 100
            st.metric("First Reply Rate", f"{first_reply_rate:.1f}%")
    
    else:
        st.success("✅ No complaint cases detected")
    
    # SERIOUS CASES
    st.markdown("---")
    st.markdown("### ⚠️ Serious Cases (Ticket Reopened)")
    
    if serious_cases:
        st.error(f"**{len(serious_cases)} serious issues** with ticket reopened")
        
        with st.expander("View Serious Cases Details", expanded=True):
            serious_data = []
            for result in serious_cases:
                serious_data.append({
                    'Ticket ID': result['ticket_id'],
                    'Main Question': result['main_question'][:60] + '...',
                    'First Reply Status': '✅ Found' if result['first_reply_found'] else '❌ Missing',
                    'Final Reply Status': '✅ Found' if result['final_reply_found'] else '❌ Missing',
                    'First Reply LT (min)': result.get('first_reply_lead_time_minutes', 'N/A'),
                    'Final Reply LT (min)': result.get('final_reply_lead_time_minutes', 'N/A'),
                    'Performance': result['performance_rating'].upper()
                })
            
            df_serious = pd.DataFrame(serious_data)
            st.dataframe(df_serious, use_container_width=True)
        
        # Serious cases analysis
        st.markdown("#### 📊 Serious Cases Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            first_reply_rate = sum(1 for r in serious_cases if r['first_reply_found']) / len(serious_cases) * 100
            st.metric("First Reply Rate", f"{first_reply_rate:.1f}%")
        
        with col2:
            final_reply_rate = sum(1 for r in serious_cases if r['final_reply_found']) / len(serious_cases) * 100
            st.metric("Final Reply Rate", f"{final_reply_rate:.1f}%")
        
        with col3:
            avg_first_lt = np.mean([r.get('first_reply_lead_time_minutes', 0) for r in serious_cases if r.get('first_reply_lead_time_minutes')])
            st.metric("Avg First Reply LT", f"{avg_first_lt:.1f} min")
    
    else:
        st.success("✅ No serious cases detected")
    
    # COMPREHENSIVE INSIGHTS
    if customer_leave_cases or complaint_cases or serious_cases:
        st.markdown("---")
        st.markdown("### 💡 Enhanced Insights & Recommendations")
        
        # Customer Leave Insights
        if customer_leave_cases:
            leave_rate = (len(customer_leave_cases) / len(successful)) * 100
            with st.expander(f"🚶 Customer Leave Insights ({leave_rate:.1f}% rate)"):
                st.write("**Issue:** Customers leaving conversations detected by Ticket Automation")
                st.write("**Recommendations:**")
                st.write("- Improve response times")
                st.write("- Implement proactive engagement")
                st.write("- Analyze reasons for customer departure")
        
        # Complaint Insights
        if complaint_cases:
            complaint_rate = (len(complaint_cases) / len(successful)) * 100
            with st.expander(f"📋 Complaint System Insights ({complaint_rate:.1f}% rate)"):
                st.write("**Observation:** Cases successfully matched from complaint system")
                st.write("**Benefits:**")
                st.write("- Accurate resolution time tracking (in days)")
                st.write("- Comprehensive case handling")
                st.write("- Better customer satisfaction monitoring")
        
        # Serious Cases Insights
        if serious_cases:
            serious_rate = (len(serious_cases) / len(successful)) * 100
            with st.expander(f"⚠️ Serious Cases Insights ({serious_rate:.1f}% rate)"):
                st.write("**Observation:** Tickets requiring reopening and additional handling")
                st.write("**Process:**")
                st.write("- First reply with action keywords")
                st.write("- Ticket reopened for further handling")
                st.write("- Final resolution after reopening")
    
    else:
        st.success("🎉 Excellent! No special handling required for these conversations.")

def display_raw_data_tab(results):
    """Display raw data tab"""
    st.markdown("## 📋 All Analyzed Data")
    
    successful = [r for r in results if r['status'] == 'success']
    
    if successful:
        st.info(f"Showing {len(successful)} successful analyses")
        
        # Tampilkan data lengkap
        raw_data = []
        for result in successful:
            raw_data.append({
                'Ticket ID': result['ticket_id'],
                'Main Question': result['main_question'],
                'Issue Type': result['final_issue_type'],
                'First Reply Found': result['first_reply_found'],
                'First Reply Message': result.get('first_reply_message', '')[:100] + '...' if result.get('first_reply_message') else 'Not found',
                'First Reply LT (min)': result.get('first_reply_lead_time_minutes'),
                'Final Reply Found': result['final_reply_found'],
                'Final Reply Message': result.get('final_reply_message', '')[:100] + '...' if result.get('final_reply_message') else 'Not found',
                'Final Reply LT': self._format_lead_time(result),
                'Performance': result['performance_rating'],
                'Quality Score': result['quality_score'],
                'Customer Leave': result['customer_leave'],
                'Total Messages': result['total_messages'],
                'Total QA Pairs': result['total_qa_pairs']
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
        
        Please ensure:
        1. File `chat_analyzer.py` exists in the same directory
        2. All dependencies are installed
        """)
        st.stop()
    
    # Check if analysis is complete
    if st.session_state.get('analysis_complete', False):
        display_enhanced_results()
    else:
        main_interface()
