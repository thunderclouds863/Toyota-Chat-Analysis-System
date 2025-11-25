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
    from Chat_Analyzer_System import (
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
    """Run enhanced analysis dengan kedua file - FIXED VERSION"""
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
        
        # Step 2: Preprocess Data
        status_text.text("🔄 Preprocessing conversation data...")
        progress_bar.progress(40)
        
        preprocessor = DataPreprocessor()
        processed_df = preprocessor.clean_data(raw_df)
        
        # Step 3: Initialize Pipeline dengan Data yang Sudah Di-load
        status_text.text("🔧 Initializing enhanced analysis pipeline...")
        progress_bar.progress(60)
        
        # PERBAIKAN: Simpan complaint file temporary dan initialize pipeline
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_complaint:
            complaint_df.to_excel(tmp_complaint.name, index=False)
            pipeline = CompleteAnalysisPipeline(complaint_data_path=tmp_complaint.name)
        
        # Step 4: Run Analysis - PERBAIKAN: Pass processed_df yang sudah benar
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
                'Final LT': _format_lead_time(result),  # PERBAIKAN: panggil function biasa
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

def _format_lead_time(result):
    """Format lead time berdasarkan jenis issue - PERBAIKAN: function biasa"""
    if result['final_issue_type'] == 'complaint' and result.get('final_reply_lead_time_days'):
        return f"{result['final_reply_lead_time_days']} days"
    elif result.get('final_reply_lead_time_minutes'):
        # Convert minutes to days jika lebih dari 1 hari
        minutes = result['final_reply_lead_time_minutes']
        if minutes > 1440:  # 24 jam * 60 menit
            days = minutes / 1440
            return f"{days:.1f} days"
        elif minutes > 60:  # lebih dari 1 jam
            hours = minutes / 60
            return f"{hours:.1f} hours"
        else:
            return f"{minutes:.1f} min"
    else:
        return "N/A"

def display_enhanced_special_cases_tab(results, stats):
    """Display enhanced special cases analysis"""
    st.markdown("## 🚨 Enhanced Special Cases Analysis")
    
    successful = [r for r in results if r['status'] == 'success']
    
    # Extract special cases - PERBAIKAN: Hanya yang benar-benar customer leave
    customer_leave_cases = [r for r in successful if r.get('customer_leave')]
    
    # Cases dengan keyword customer leave tapi punya replies
    false_leave_cases = []
    for r in successful:
        # Cek apakah ada keyword customer leave di raw data
        has_leave_keyword = False
        if '_raw_qa_pairs' in r:
            for qa_pair in r['_raw_qa_pairs']:
                message = qa_pair.get('answer', '')
                if config.CUSTOMER_LEAVE_KEYWORD in str(message):
                    has_leave_keyword = True
                    break
        
        # Jika ada keyword tapi tidak di-mark sebagai customer leave, berarti punya replies
        if has_leave_keyword and not r.get('customer_leave'):
            false_leave_cases.append(r)
    
    complaint_cases = [r for r in successful if r['final_issue_type'] == 'complaint']
    serious_cases = [r for r in successful if r['final_issue_type'] == 'serious']
    
    # SUMMARY CARDS
    st.markdown("### 📊 Special Cases Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("True Customer Leave", len(customer_leave_cases))
    
    with col2:
        st.metric("False Customer Leave", len(false_leave_cases))
    
    with col3:
        st.metric("Complaint Cases", len(complaint_cases))
    
    with col4:
        st.metric("Serious Cases", len(serious_cases))
    
    # TRUE CUSTOMER LEAVE CASES
    st.markdown("---")
    st.markdown("### 🚶 True Customer Leave Cases (No Proper Replies)")
    
    if customer_leave_cases:
        st.error(f"**{len(customer_leave_cases)} conversations** where customer left WITHOUT proper replies")
        
        with st.expander("View True Customer Leave Details", expanded=True):
            leave_data = []
            for result in customer_leave_cases:
                leave_data.append({
                    'Ticket ID': result['ticket_id'],
                    'Issue Type': result['final_issue_type'].upper(),
                    'Main Question': result['main_question'][:60] + '...',
                    'First Reply': '❌ Missing' if not result['first_reply_found'] else '⚠️ Partial',
                    'Final Reply': '❌ Missing' if not result['final_reply_found'] else '⚠️ Partial',
                    'Performance': result['performance_rating'].upper()
                })
            
            df_leave = pd.DataFrame(leave_data)
            st.dataframe(df_leave, use_container_width=True)
    else:
        st.success("✅ No true customer leave cases detected")
    
    # FALSE CUSTOMER LEAVE CASES (NEW SECTION)
    if false_leave_cases:
        st.markdown("---")
        st.markdown("### ⚠️ False Customer Leave Cases (Has Replies)")
        
        st.warning(f"**{len(false_leave_cases)} conversations** with customer leave keyword BUT proper replies found")
        
        with st.expander("View False Customer Leave Details", expanded=True):
            false_leave_data = []
            for result in false_leave_cases:
                false_leave_data.append({
                    'Ticket ID': result['ticket_id'],
                    'Issue Type': result['final_issue_type'].upper(),
                    'Main Question': result['main_question'][:60] + '...',
                    'First Reply': '✅ Found' if result['first_reply_found'] else '❌ Missing',
                    'Final Reply': '✅ Found' if result['final_reply_found'] else '❌ Missing',
                    'Performance': result['performance_rating'].upper(),
                    'Note': 'Replies found despite leave keyword'
                })
            
            df_false_leave = pd.DataFrame(false_leave_data)
            st.dataframe(df_false_leave, use_container_width=True)

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
            # PERBAIKAN: Handle complaint cases yang tidak ada final_reply_avg_minutes
            if lt_stats['final_reply_avg_minutes'] > 0 and lt_stats['final_reply_avg_minutes'] != float('inf'):
                st.metric("Final Reply Average", f"{lt_stats['final_reply_avg_minutes']:.1f} min")
            else:
                st.metric("Final Reply Average", "Mixed (min/days)")
        with col3:
            st.metric("First Reply Samples", lt_stats['first_reply_samples'])
        with col4:
            st.metric("Final Reply Samples", lt_stats['final_reply_samples'])
    
    # Lead Time by Issue Type - PERBAIKAN DENGAN VALIDASI KETAT
    st.markdown("### 📈 Lead Time by Issue Type")
    
    lead_time_by_type = {}
    for result in successful:
        issue_type = result['final_issue_type']
        if issue_type not in lead_time_by_type:
            lead_time_by_type[issue_type] = {
                'first_lead_times': [],
                'final_lead_minutes': [],
                'final_lead_days': []  # Untuk complaint
            }
        
        # First reply lead times (selalu dalam minutes) - DENGAN VALIDASI
        first_lt = result.get('first_reply_lead_time_minutes')
        if first_lt is not None and first_lt != 'N/A':
            try:
                first_lt_float = float(first_lt)
                if first_lt_float > 0:  # Hanya ambil nilai positif
                    lead_time_by_type[issue_type]['first_lead_times'].append(first_lt_float)
            except (ValueError, TypeError):
                pass  # Skip jika tidak bisa di-convert
        
        # Final reply lead times - PERBAIKAN DENGAN VALIDASI KETAT
        if issue_type == 'complaint':
            final_lt_days = result.get('final_reply_lead_time_days')
            if final_lt_days is not None and final_lt_days != 'N/A':
                try:
                    final_lt_days_float = float(final_lt_days)
                    if final_lt_days_float > 0:  # Hanya ambil nilai positif
                        lead_time_by_type[issue_type]['final_lead_days'].append(final_lt_days_float)
                except (ValueError, TypeError):
                    pass  # Skip jika tidak bisa di-convert
        else:
            final_lt_min = result.get('final_reply_lead_time_minutes')
            if final_lt_min is not None and final_lt_min != 'N/A':
                try:
                    final_lt_min_float = float(final_lt_min)
                    if final_lt_min_float > 0:  # Hanya ambil nilai positif
                        lead_time_by_type[issue_type]['final_lead_minutes'].append(final_lt_min_float)
                except (ValueError, TypeError):
                    pass  # Skip jika tidak bisa di-convert
    
    # Display lead time by type - PERBAIKAN DENGAN ERROR HANDLING
    lead_time_data = []
    for issue_type, data in lead_time_by_type.items():
        # First reply average (selalu minutes)
        first_avg = None
        if data['first_lead_times']:
            try:
                first_avg = np.mean(data['first_lead_times'])
            except:
                first_avg = None
        
        # Final reply average - PERBAIKAN DENGAN ERROR HANDLING
        final_avg = None
        final_unit = 'N/A'
        final_samples = 0
        
        if issue_type == 'complaint' and data['final_lead_days']:
            try:
                final_avg = np.mean(data['final_lead_days'])
                final_unit = 'days'
                final_samples = len(data['final_lead_days'])
            except:
                final_avg = None
        elif data['final_lead_minutes']:
            try:
                final_avg = np.mean(data['final_lead_minutes'])
                final_unit = 'min'
                final_samples = len(data['final_lead_minutes'])
            except:
                final_avg = None
        
        lead_time_data.append({
            'Issue Type': issue_type.upper(),
            'First Reply Avg': f"{first_avg:.1f} min" if first_avg else 'N/A',
            'Final Reply Avg': f"{final_avg:.1f} {final_unit}" if final_avg else 'N/A',
            'First Reply Samples': len(data['first_lead_times']),
            'Final Reply Samples': final_samples
        })
    
    if lead_time_data:
        df_lead_time = pd.DataFrame(lead_time_data)
        st.dataframe(df_lead_time, use_container_width=True)
    else:
        st.info("No lead time data available")
    
    # Lead Time Distribution - PERBAIKAN: Filter hanya normal/serious untuk minutes
    st.markdown("### 📊 Lead Time Distribution (Normal & Serious Cases)")
    
    # PERBAIKAN: Validasi data sebelum plotting
    normal_final_times = []
    serious_final_times = []
    
    for r in successful:
        if r['final_issue_type'] == 'normal' and r.get('final_reply_lead_time_minutes'):
            try:
                lt = float(r['final_reply_lead_time_minutes'])
                if lt > 0:
                    normal_final_times.append(lt)
            except:
                pass
        
        if r['final_issue_type'] == 'serious' and r.get('final_reply_lead_time_minutes'):
            try:
                lt = float(r['final_reply_lead_time_minutes'])
                if lt > 0:
                    serious_final_times.append(lt)
            except:
                pass
    
    if normal_final_times or serious_final_times:
        col1, col2 = st.columns(2)
        
        with col1:
            # Normal issues lead time
            if normal_final_times:
                try:
                    fig_normal = px.histogram(
                        x=normal_final_times,
                        title='Normal Issues - Final Reply Lead Time',
                        labels={'x': 'Lead Time (minutes)', 'y': 'Frequency'},
                        nbins=20,
                        color_discrete_sequence=['#2E86AB']
                    )
                    mean_normal = np.mean(normal_final_times)
                    fig_normal.add_vline(
                        x=mean_normal, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text=f"Mean: {mean_normal:.1f} min"
                    )
                    st.plotly_chart(fig_normal, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating normal issues chart: {e}")
        
        with col2:
            # Serious issues lead time
            if serious_final_times:
                try:
                    fig_serious = px.histogram(
                        x=serious_final_times,
                        title='Serious Issues - Final Reply Lead Time',
                        labels={'x': 'Lead Time (minutes)', 'y': 'Frequency'},
                        nbins=20,
                        color_discrete_sequence=['#A23B72']
                    )
                    mean_serious = np.mean(serious_final_times)
                    fig_serious.add_vline(
                        x=mean_serious, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text=f"Mean: {mean_serious:.1f} min"
                    )
                    st.plotly_chart(fig_serious, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating serious issues chart: {e}")
    else:
        st.info("No lead time data available for distribution analysis")
    
    # Complaint Cases Summary - TABEL BARU untuk complaint
    complaint_cases = [r for r in successful if r['final_issue_type'] == 'complaint']
    if complaint_cases:
        st.markdown("### 📋 Complaint Cases Resolution Time (Days)")
        
        complaint_data = []
        complaint_lead_times = []
        
        for result in complaint_cases:
            lead_time_days = result.get('final_reply_lead_time_days', 'N/A')
            
            # Validasi dan kumpulkan data untuk statistics
            if lead_time_days not in [None, 'N/A']:
                try:
                    lt_days = float(lead_time_days)
                    complaint_lead_times.append(lt_days)
                    display_lt = f"{lt_days} days"
                except (ValueError, TypeError):
                    display_lt = 'N/A'
            else:
                display_lt = 'N/A'
            
            complaint_data.append({
                'Ticket ID': result['ticket_id'],
                'Main Question': result['main_question'][:60] + '...',
                'Resolution Time': display_lt,
                'First Reply Found': '✅' if result['first_reply_found'] else '❌'
            })
        
        df_complaint = pd.DataFrame(complaint_data)
        st.dataframe(df_complaint, use_container_width=True)
        
        # Complaint statistics - HANYA jika ada data valid
        if complaint_lead_times:
            st.markdown("#### 📊 Complaint Resolution Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                try:
                    avg_days = np.mean(complaint_lead_times)
                    st.metric("Avg Resolution Time", f"{avg_days:.1f} days")
                except:
                    st.metric("Avg Resolution Time", "N/A")
            with col2:
                try:
                    median_days = np.median(complaint_lead_times)
                    st.metric("Median Resolution Time", f"{median_days:.1f} days")
                except:
                    st.metric("Median Resolution Time", "N/A")
            with col3:
                st.metric("Complaint Cases", len(complaint_cases))

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
                'Final Reply LT': _format_lead_time(result),  # PERBAIKAN: panggil function biasa
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
            complaint_lead_times_days = []  # PERBAIKAN: Kumpulkan data lead time
            
            for result in complaint_cases:
                lead_time_days = result.get('final_reply_lead_time_days', 'N/A')
                
                # Validasi lead time
                if lead_time_days not in [None, 'N/A']:
                    try:
                        lt_days = float(lead_time_days)
                        complaint_lead_times_days.append(lt_days)
                        display_lt = f"{lt_days} days"
                    except (ValueError, TypeError):
                        display_lt = 'N/A'
                else:
                    display_lt = 'N/A'
                
                complaint_data.append({
                    'Ticket ID': result['ticket_id'],
                    'Main Question': result['main_question'][:60] + '...',
                    'First Reply Status': '✅ Found' if result['first_reply_found'] else '❌ Missing',
                    'Final Resolution Time': display_lt,
                    'Performance': result['performance_rating'].upper()
                })
            
            df_complaint = pd.DataFrame(complaint_data)
            st.dataframe(df_complaint, use_container_width=True)
        
        # Complaint analysis - PERBAIKAN: dengan validasi
        st.markdown("#### 📈 Complaint Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            if complaint_lead_times_days:
                try:
                    avg_lead_time = np.mean(complaint_lead_times_days)
                    st.metric("Avg Resolution Time", f"{avg_lead_time:.1f} days")
                except:
                    st.metric("Avg Resolution Time", "N/A")
            else:
                st.metric("Avg Resolution Time", "N/A")
        
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
            serious_lead_times_days = []  # PERBAIKAN: Konversi minutes ke days
            
            for result in serious_cases:
                # Konversi final reply lead time dari minutes ke days
                final_lt_min = result.get('final_reply_lead_time_minutes')
                final_lt_days = None
                
                if final_lt_min not in [None, 'N/A']:
                    try:
                        final_lt_min_float = float(final_lt_min)
                        final_lt_days = final_lt_min_float / (24 * 60)  # Convert minutes to days
                        serious_lead_times_days.append(final_lt_days)
                        display_final_lt = f"{final_lt_days:.2f} days"
                    except (ValueError, TypeError):
                        display_final_lt = 'N/A'
                else:
                    display_final_lt = 'N/A'
                
                # First reply lead time (tetap dalam minutes untuk display)
                first_lt_min = result.get('first_reply_lead_time_minutes', 'N/A')
                display_first_lt = f"{first_lt_min} min" if first_lt_min not in [None, 'N/A'] else 'N/A'
                
                serious_data.append({
                    'Ticket ID': result['ticket_id'],
                    'Main Question': result['main_question'][:60] + '...',
                    'First Reply Status': '✅ Found' if result['first_reply_found'] else '❌ Missing',
                    'Final Reply Status': '✅ Found' if result['final_reply_found'] else '❌ Missing',
                    'First Reply LT': display_first_lt,
                    'Final Reply LT': display_final_lt,
                    'Performance': result['performance_rating'].upper()
                })
            
            df_serious = pd.DataFrame(serious_data)
            st.dataframe(df_serious, use_container_width=True)
        
        # Serious cases analysis - PERBAIKAN: dengan data yang sudah dikonversi
        st.markdown("#### 📊 Serious Cases Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            first_reply_rate = sum(1 for r in serious_cases if r['first_reply_found']) / len(serious_cases) * 100
            st.metric("First Reply Rate", f"{first_reply_rate:.1f}%")
        
        with col2:
            final_reply_rate = sum(1 for r in serious_cases if r['final_reply_found']) / len(serious_cases) * 100
            st.metric("Final Reply Rate", f"{final_reply_rate:.1f}%")
        
        with col3:
            if serious_lead_times_days:
                try:
                    avg_final_lt_days = np.mean(serious_lead_times_days)
                    st.metric("Avg Final Reply LT", f"{avg_final_lt_days:.2f} days")
                except:
                    st.metric("Avg Final Reply LT", "N/A")
            else:
                st.metric("Avg Final Reply LT", "N/A")
    
    else:
        st.success("✅ No serious cases detected")
    
    # OVERALL LEAD TIME SUMMARY (SEMUA ISSUE TYPES) - FITUR BARU
    st.markdown("---")
    st.markdown("### 📊 Overall Final Reply Lead Time Summary (All in Days)")
    
    # Kumpulkan semua final reply lead times dalam days
    all_final_lead_times_days = []
    
    for result in successful:
        if result['final_issue_type'] == 'complaint':
            # Complaint: langsung ambil days
            lt_days = result.get('final_reply_lead_time_days')
            if lt_days not in [None, 'N/A']:
                try:
                    all_final_lead_times_days.append(float(lt_days))
                except (ValueError, TypeError):
                    pass
        else:
            # Normal & Serious: convert minutes to days
            lt_min = result.get('final_reply_lead_time_minutes')
            if lt_min not in [None, 'N/A']:
                try:
                    lt_min_float = float(lt_min)
                    lt_days = lt_min_float / (24 * 60)  # Convert minutes to days
                    all_final_lead_times_days.append(lt_days)
                except (ValueError, TypeError):
                    pass
    
    if all_final_lead_times_days:
        st.markdown("#### 📈 Overall Statistics (Final Reply Lead Time)")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            try:
                avg_days = np.mean(all_final_lead_times_days)
                st.metric("Average", f"{avg_days:.2f} days")
            except:
                st.metric("Average", "N/A")
        
        with col2:
            try:
                median_days = np.median(all_final_lead_times_days)
                st.metric("Median", f"{median_days:.2f} days")
            except:
                st.metric("Median", "N/A")
        
        with col3:
            try:
                min_days = np.min(all_final_lead_times_days)
                st.metric("Minimum", f"{min_days:.2f} days")
            except:
                st.metric("Minimum", "N/A")
        
        with col4:
            try:
                max_days = np.max(all_final_lead_times_days)
                st.metric("Maximum", f"{max_days:.2f} days")
            except:
                st.metric("Maximum", "N/A")
        
        # Distribution chart
        st.markdown("#### 📊 Distribution of Final Reply Lead Times")
        fig_all = px.histogram(
            x=all_final_lead_times_days,
            title='All Issues - Final Reply Lead Time Distribution (Days)',
            labels={'x': 'Lead Time (days)', 'y': 'Frequency'},
            nbins=20,
            color_discrete_sequence=['#2E86AB']
        )
        try:
            mean_all = np.mean(all_final_lead_times_days)
            fig_all.add_vline(
                x=mean_all, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Mean: {mean_all:.2f} days"
            )
        except:
            pass
        st.plotly_chart(fig_all, use_container_width=True)
    else:
        st.info("No final reply lead time data available for overall analysis")
    
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
                'Final Reply LT': _format_lead_time(result),  # PERBAIKAN: panggil function biasa
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







