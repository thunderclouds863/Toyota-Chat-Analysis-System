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
    st.markdown('<h1 class="main-header">🤖 Live Chat Analysis</h1>', unsafe_allow_html=True)
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
            min_value=0,
            max_value=100000,
            value=100,
            help="Limit number of tickets for faster analysis"
        )
        
        st.markdown("---")
        
        # Analysis Button
        if uploaded_raw_file is not None and uploaded_complaint_file is not None:
            if st.button("🚀 START ANALYSIS", type="primary", use_container_width=True):
                with st.spinner("🔄 Starting analysis..."):
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
        **Analysis Features:**
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
        st.markdown("## 🚀 Ready for Analysis")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🎯 Start Analysis", type="primary", use_container_width=True):
                with st.spinner("🔄 Starting analysis..."):
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
        status_text.text("🔧 Initializing analysis pipeline...")
        progress_bar.progress(60)
        
        # PERBAIKAN: Simpan complaint file temporary dan initialize pipeline
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_complaint:
            complaint_df.to_excel(tmp_complaint.name, index=False)
            pipeline = CompleteAnalysisPipeline(complaint_data_path=tmp_complaint.name)
        
        # Step 4: Run Analysis - PERBAIKAN: Pass processed_df yang sudah benar
        status_text.text("🔍 Analyzing conversations with logic...")
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
        status_text.text("✅ analysis complete!")
        
        return results, stats, excel_path
        
    except Exception as e:
        st.error(f"❌ analysis error: {str(e)}")
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
    st.markdown('<h1 class="main-header">📊 Live Chat Performance Dashboard</h1>', unsafe_allow_html=True)
    
    # QUICK OVERVIEW METRICS - PROFESSIONAL STYLE
    st.markdown("## 📈 Executive Summary")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        total_tickets = stats.get('total_tickets', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Conversations</h3>
            <h1>{total_tickets}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        success_rate = stats.get('success_rate', 0) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>Analysis Success Rate</h3>
            <h1>{success_rate:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        if 'reply_effectiveness' in stats:
            final_reply_rate = stats['reply_effectiveness'].get('final_reply_found_rate', 0) * 100
        else:
            final_reply_rate = 0
            
        st.markdown(f"""
        <div class="metric-card">
            <h3>Final Reply Rate</h3>
            <h1>{final_reply_rate:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    # DOWNLOAD SECTION
    st.markdown("---")
    st.markdown("## 💾 Download Complete Analysis Report")
    
    if st.session_state.get('excel_file_path') and os.path.exists(st.session_state.excel_file_path):
        with open(st.session_state.excel_file_path, "rb") as f:
            excel_data = f.read()
        
        st.download_button(
            label="📥 DOWNLOAD FULL EXCEL REPORT",
            data=excel_data,
            file_name=f"chat_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True
        )
        st.success("✅ Report contains detailed analysis, lead times, performance metrics, and raw data")
    else:
        st.error("❌ Excel file not available for download")

    # TABS - PROFESSIONAL LAYOUT
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "🎯 Issue Types", "⏱️ Lead Times", "📈 Performance", "🚨 Special Cases", "🔍 Raw Data"
    ])
    
    with tab1:
        display_professional_overview_tab(results, stats)
    
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

def display_professional_overview_tab(results, stats):
    """Display professional overview tab dengan rangkuman penting"""
    st.markdown("## 📊 Performance Overview")
    
    # ROW 1: Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Issue Type Distribution
        if 'issue_type_distribution' in stats:
            total_issues = sum(stats['issue_type_distribution'].values())
            normal_pct = (stats['issue_type_distribution'].get('normal', 0) / total_issues * 100) if total_issues > 0 else 0
            st.metric("Normal Inquiries", f"{normal_pct:.1f}%")
    
    with col2:
        if 'issue_type_distribution' in stats:
            total_issues = sum(stats['issue_type_distribution'].values())
            serious_pct = (stats['issue_type_distribution'].get('serious', 0) / total_issues * 100) if total_issues > 0 else 0
            st.metric("Serious Cases", f"{serious_pct:.1f}%")
    
    with col3:
        if 'issue_type_distribution' in stats:
            total_issues = sum(stats['issue_type_distribution'].values())
            complaint_pct = (stats['issue_type_distribution'].get('complaint', 0) / total_issues * 100) if total_issues > 0 else 0
            st.metric("Complaint Cases", f"{complaint_pct:.1f}%")
    
    with col4:
        if 'reply_effectiveness' in stats:
            customer_leave = stats['reply_effectiveness'].get('customer_leave_cases', 0)
            total_successful = stats.get('successful_analysis', 1)
            leave_rate = (customer_leave / total_successful * 100) if total_successful > 0 else 0
            st.metric("Customer Leave Rate", f"{leave_rate:.1f}%")
    
    # ROW 2: Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Issue Type Distribution Pie Chart
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
                values=counts, 
                names=issue_types,
                title='Issue Type Distribution',
                color=issue_types,
                color_discrete_map=color_map
            )
            fig_issues.update_layout(showlegend=True)
            st.plotly_chart(fig_issues, use_container_width=True)
    
    with col2:
        # Performance Distribution
        if 'performance_distribution' in stats:
            performances = list(stats['performance_distribution'].keys())
            counts = list(stats['performance_distribution'].values())
            
            # Sort performances dalam order yang meaningful
            performance_order = ['excellent', 'good', 'fair', 'poor']
            performances_sorted = [p for p in performance_order if p in performances]
            counts_sorted = [counts[performances.index(p)] for p in performances_sorted]
            
            fig_perf = px.bar(
                x=performances_sorted, 
                y=counts_sorted,
                title='Performance Rating Distribution',
                labels={'x': 'Performance Rating', 'y': 'Number of Conversations'},
                color=performances_sorted,
                color_discrete_map={
                    'excellent': '#28a745',
                    'good': '#17a2b8', 
                    'fair': '#ffc107',
                    'poor': '#dc3545'
                }
            )
            fig_perf.update_layout(xaxis_title="Performance Rating", yaxis_title="Count")
            st.plotly_chart(fig_perf, use_container_width=True)
    
    # ROW 3: Lead Time Summary
    st.markdown("### ⏱️ Lead Time Performance")
    
    if 'lead_time_stats' in stats:
        lt_stats = stats['lead_time_stats']
        st.markdown("#### ⏱️ Lead Time Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            first_avg = lt_stats.get('first_reply_lead_time_minutes', 0)
            if first_avg > 1440:  # lebih dari 1 hari
                display_first = f"{first_avg/1440:.1f} days"
            elif first_avg > 60:  # lebih dari 1 jam
                display_first = f"{first_avg/60:.1f} hours"
            else:
                display_first = f"{first_avg:.0f} min"
            st.metric("First Reply Avg", display_first)
        
        with col2:
            final_avg = lt_stats.get('final_reply_avg_minutes', 0)
            if final_avg > 1440:  # lebih dari 1 hari
                display_final = f"{final_avg/1440:.1f} days"
            elif final_avg > 60:  # lebih dari 1 jam
                display_final = f"{final_avg/60:.1f} hours"
            else:
                display_final = f"{final_avg:.0f} min"
            st.metric("Final Reply Avg", display_final)
        
        with col3:
            st.metric("First Reply Samples", lt_stats['first_reply_samples'])
        
        with col4:
            st.metric("Final Reply Samples", lt_stats['final_reply_samples'])
    
    # ROW 4: Reply Effectiveness
    st.markdown("### 💬 Reply Effectiveness")
    
    if 'reply_effectiveness' in stats:
        eff = stats['reply_effectiveness']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            first_reply_rate = eff.get('first_reply_found_rate', 0) * 100
            st.metric("First Reply Found", f"{first_reply_rate:.1f}%")
        
        with col2:
            final_reply_rate = eff.get('final_reply_found_rate', 0) * 100
            st.metric("Final Reply Found", f"{final_reply_rate:.1f}%")
        
        with col3:
            customer_leave = eff.get('customer_leave_cases', 0)
            st.metric("Customer Leave Cases", customer_leave)

def display_enhanced_lead_time_tab(results, stats):
    """Display enhanced lead time analysis - SEMUA ISSUE TYPE DISATUKAN"""
    st.markdown("## ⏱️ Lead Time Analysis")
    
    successful = [r for r in results if r['status'] == 'success']
    
    # Helper function untuk format lead time yang fleksibel
    def format_lead_time(minutes):
        """Format lead time berdasarkan durasi: minutes, hours, atau days"""
        if minutes is None or minutes == 'N/A':
            return "N/A"
        
        try:
            minutes_float = float(minutes)
            if minutes_float <= 0:
                return "N/A"
            
            if minutes_float > 1440:  # > 1 day
                days = minutes_float / 1440
                return f"{days:.1f} days"
            elif minutes_float > 60:  # > 1 hour
                hours = minutes_float / 60
                return f"{hours:.1f} hours"
            else:
                return f"{minutes_float:.1f} min"
        except (ValueError, TypeError):
            return "N/A"
    
    # 1. SUMMARY TOTAL - Convert semua ke menit dulu
    st.markdown("### 📊 Overall Lead Time Summary")
    
    # Hitung average total untuk semua issue types (semua dalam minutes)
    all_first_lead_times = []
    all_final_lead_times_minutes = []
    
    for result in successful:
        # First reply lead times (selalu dalam minutes)
        first_lt = result.get('first_reply_lead_time_minutes')
        if first_lt is not None and first_lt != 'N/A':
            try:
                first_lt_float = float(first_lt)
                if first_lt_float > 0:
                    all_first_lead_times.append(first_lt_float)
            except (ValueError, TypeError):
                pass
        
        # Final reply lead times - Convert SEMUA ke minutes
        final_lt_minutes = None
        if result['final_issue_type'] == 'complaint':
            # Complaint: convert days ke minutes
            final_lt_days = result.get('final_reply_lead_time_days')
            if final_lt_days is not None and final_lt_days != 'N/A':
                try:
                    final_lt_minutes = float(final_lt_days) * 24 * 60  # Convert days to minutes
                except (ValueError, TypeError):
                    pass
        else:
            # Normal & Serious: langsung ambil minutes
            final_lt_min = result.get('final_reply_lead_time_minutes')
            if final_lt_min is not None and final_lt_min != 'N/A':
                try:
                    final_lt_minutes = float(final_lt_min)
                except (ValueError, TypeError):
                    pass
        
        if final_lt_minutes is not None and final_lt_minutes > 0:
            all_final_lead_times_minutes.append(final_lt_minutes)
    
    # Calculate averages dengan format fleksibel
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if all_first_lead_times:
            avg_first_minutes = np.mean(all_first_lead_times)
            st.metric("First Reply Average", format_lead_time(avg_first_minutes))
        else:
            st.metric("First Reply Average", "N/A")
    
    with col2:
        if all_final_lead_times_minutes:
            avg_final_minutes = np.mean(all_final_lead_times_minutes)
            st.metric("Final Reply Average", format_lead_time(avg_final_minutes))
        else:
            st.metric("Final Reply Average", "N/A")
    
    with col3:
        st.metric("First Reply Samples", len(all_first_lead_times))
    
    with col4:
        st.metric("Final Reply Samples", len(all_final_lead_times_minutes))

    # 2. BREAKDOWN PER ISSUE TYPE (SEMUA DISATUKAN)
    st.markdown("### 📈 Lead Time Breakdown by Issue Type")
    
    lead_time_by_type = {}
    for result in successful:
        issue_type = result['final_issue_type']
        if issue_type not in lead_time_by_type:
            lead_time_by_type[issue_type] = {
                'first_lead_times': [],
                'final_lead_times_minutes': []  # Semua dalam minutes
            }
        
        # First reply lead times
        first_lt = result.get('first_reply_lead_time_minutes')
        if first_lt is not None and first_lt != 'N/A':
            try:
                first_lt_float = float(first_lt)
                if first_lt_float > 0:
                    lead_time_by_type[issue_type]['first_lead_times'].append(first_lt_float)
            except (ValueError, TypeError):
                pass
        
        # Final reply lead times - Convert SEMUA ke minutes
        final_lt_minutes = None
        if issue_type == 'complaint':
            final_lt_days = result.get('final_reply_lead_time_days')
            if final_lt_days is not None and final_lt_days != 'N/A':
                try:
                    final_lt_minutes = float(final_lt_days) * 24 * 60  # Convert days to minutes
                except (ValueError, TypeError):
                    pass
        else:
            final_lt_min = result.get('final_reply_lead_time_minutes')
            if final_lt_min is not None and final_lt_min != 'N/A':
                try:
                    final_lt_minutes = float(final_lt_min)
                except (ValueError, TypeError):
                    pass
        
        if final_lt_minutes is not None and final_lt_minutes > 0:
            lead_time_by_type[issue_type]['final_lead_times_minutes'].append(final_lt_minutes)
    
    # Display breakdown table dengan format fleksibel
    breakdown_data = []
    for issue_type, data in lead_time_by_type.items():
        # First reply average
        first_avg = None
        if data['first_lead_times']:
            try:
                first_avg_minutes = np.mean(data['first_lead_times'])
                first_avg = format_lead_time(first_avg_minutes)
            except:
                first_avg = 'N/A'
        else:
            first_avg = 'N/A'
        
        # Final reply average
        final_avg = None
        if data['final_lead_times_minutes']:
            try:
                final_avg_minutes = np.mean(data['final_lead_times_minutes'])
                final_avg = format_lead_time(final_avg_minutes)
            except:
                final_avg = 'N/A'
        else:
            final_avg = 'N/A'
        
        breakdown_data.append({
            'Issue Type': issue_type.upper(),
            'First Reply Avg': first_avg,
            'Final Reply Avg': final_avg,
            'First Reply Samples': len(data['first_lead_times']),
            'Final Reply Samples': len(data['final_lead_times_minutes'])
        })
    
    if breakdown_data:
        df_breakdown = pd.DataFrame(breakdown_data)
        st.dataframe(df_breakdown, use_container_width=True)
    else:
        st.info("No lead time data available for breakdown")

    # 3. DISTRIBUTION ANALYSIS - FIRST REPLY
    st.markdown("### 📊 First Reply Lead Time Distribution - All Cases")
    
    # Kumpulkan semua first reply lead times
    first_reply_data = []
    for result in successful:
        first_lt = result.get('first_reply_lead_time_minutes')
        if first_lt is not None and first_lt != 'N/A':
            try:
                first_lt_float = float(first_lt)
                if first_lt_float > 0:
                    first_reply_data.append({
                        'Issue Type': result['final_issue_type'].upper(),
                        'Lead Time (minutes)': first_lt_float,
                        'Lead Time (hours)': first_lt_float / 60,
                        'Formatted Lead Time': format_lead_time(first_lt_float)
                    })
            except (ValueError, TypeError):
                pass
    
    if first_reply_data:
        df_first = pd.DataFrame(first_reply_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot first reply
            fig_first_box = px.box(
                df_first, 
                x='Issue Type', 
                y='Lead Time (minutes)',
                title='First Reply Lead Time Distribution (Minutes)',
                color='Issue Type',
                color_discrete_map={
                    'NORMAL': '#2E86AB',
                    'SERIOUS': '#A23B72',
                    'COMPLAINT': '#F18F01'
                }
            )
            st.plotly_chart(fig_first_box, use_container_width=True)
        
        with col2:
            # Statistics untuk first reply
            st.markdown("#### 📈 First Reply Statistics")
            first_stats_data = []
            for issue_type in df_first['Issue Type'].unique():
                issue_data = df_first[df_first['Issue Type'] == issue_type]['Lead Time (minutes)']
                if len(issue_data) > 0:
                    first_stats_data.append({
                        'Issue Type': issue_type,
                        'Count': len(issue_data),
                        'Average': format_lead_time(np.mean(issue_data)),
                        'Median': format_lead_time(np.median(issue_data)),
                        'Min': format_lead_time(np.min(issue_data)),
                        'Max': format_lead_time(np.max(issue_data))
                    })
            
            if first_stats_data:
                df_first_stats = pd.DataFrame(first_stats_data)
                st.dataframe(df_first_stats, use_container_width=True)
    else:
        st.info("No first reply lead time data available")

    # 4. DISTRIBUTION ANALYSIS - FINAL REPLY
    st.markdown("### 📊 Final Reply Lead Time Distribution - All Cases")
    
    # Kumpulkan semua final reply lead times dalam minutes
    final_reply_data = []
    for result in successful:
        # Convert semua final reply ke minutes
        final_lt_minutes = None
        
        if result['final_issue_type'] == 'complaint':
            # Complaint: convert days ke minutes
            final_lt_days = result.get('final_reply_lead_time_days')
            if final_lt_days is not None and final_lt_days != 'N/A':
                try:
                    final_lt_minutes = float(final_lt_days) * 24 * 60  # Convert days to minutes
                except (ValueError, TypeError):
                    pass
        else:
            # Normal & Serious: langsung ambil minutes
            final_lt_min = result.get('final_reply_lead_time_minutes')
            if final_lt_min is not None and final_lt_min != 'N/A':
                try:
                    final_lt_minutes = float(final_lt_min)
                except (ValueError, TypeError):
                    pass
        
        if final_lt_minutes is not None and final_lt_minutes > 0:
            final_reply_data.append({
                'Issue Type': result['final_issue_type'].upper(),
                'Lead Time (minutes)': final_lt_minutes,
                'Lead Time (hours)': final_lt_minutes / 60,
                'Lead Time (days)': final_lt_minutes / 1440,
                'Formatted Lead Time': format_lead_time(final_lt_minutes)
            })
    
    if final_reply_data:
        df_final = pd.DataFrame(final_reply_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot final reply - pilih unit yang paling appropriate
            # Jika ada data > 1 day, gunakan days, else gunakan hours
            max_lead_time = df_final['Lead Time (days)'].max()
            if max_lead_time > 1:
                y_column = 'Lead Time (days)'
                y_title = 'Lead Time (days)'
                title_suffix = 'Days'
            else:
                y_column = 'Lead Time (hours)'
                y_title = 'Lead Time (hours)'
                title_suffix = 'Hours'
            
            fig_final_box = px.box(
                df_final, 
                x='Issue Type', 
                y=y_column,
                title=f'Final Reply Lead Time Distribution ({title_suffix})',
                color='Issue Type',
                color_discrete_map={
                    'NORMAL': '#2E86AB',
                    'SERIOUS': '#A23B72',
                    'COMPLAINT': '#F18F01'
                }
            )
            fig_final_box.update_layout(yaxis_title=y_title)
            st.plotly_chart(fig_final_box, use_container_width=True)
        
        with col2:
            # Statistics untuk final reply dengan format fleksibel
            st.markdown("#### 📈 Final Reply Statistics")
            final_stats_data = []
            for issue_type in df_final['Issue Type'].unique():
                issue_data = df_final[df_final['Issue Type'] == issue_type]['Lead Time (minutes)']
                if len(issue_data) > 0:
                    final_stats_data.append({
                        'Issue Type': issue_type,
                        'Count': len(issue_data),
                        'Average': format_lead_time(np.mean(issue_data)),
                        'Median': format_lead_time(np.median(issue_data)),
                        'Min': format_lead_time(np.min(issue_data)),
                        'Max': format_lead_time(np.max(issue_data))
                    })
            
            if final_stats_data:
                df_final_stats = pd.DataFrame(final_stats_data)
                st.dataframe(df_final_stats, use_container_width=True)
        
        # Additional insight untuk final reply
        st.markdown("#### 💡 Final Reply Insights")
        total_final_cases = len(final_reply_data)
        if total_final_cases > 0:
            st.write(f"**Total final reply cases analyzed:** {total_final_cases}")
            
            # Cari issue type dengan lead time terpanjang
            max_lead_time_idx = df_final['Lead Time (minutes)'].idxmax()
            max_case = df_final.loc[max_lead_time_idx]
            st.write(f"**Longest resolution:** {max_case['Issue Type']} case - {max_case['Formatted Lead Time']}")
            
            # Cari issue type dengan lead time terpendek
            min_lead_time_idx = df_final['Lead Time (minutes)'].idxmin()
            min_case = df_final.loc[min_lead_time_idx]
            st.write(f"**Fastest resolution:** {min_case['Issue Type']} case - {min_case['Formatted Lead Time']}")
            
            # Rata-rata overall
            overall_avg = format_lead_time(df_final['Lead Time (minutes)'].mean())
            st.write(f"**Overall average resolution time:** {overall_avg}")
    else:
        st.info("No final reply lead time data available")

    # 5. COMPARISON CHART (SEMUA ISSUE TYPE)
    st.markdown("### 📊 Lead Time Comparison - All Issue Types")
    
    # Prepare data untuk comparison chart
    comparison_data = []
    for issue_type, data in lead_time_by_type.items():
        if data['first_lead_times']:
            try:
                first_avg = np.mean(data['first_lead_times'])
                comparison_data.append({
                    'Issue Type': issue_type.upper(),
                    'Reply Type': 'First Reply',
                    'Average Lead Time (minutes)': first_avg,
                    'Formatted Time': format_lead_time(first_avg)
                })
            except:
                pass
        
        if data['final_lead_times_minutes']:
            try:
                final_avg = np.mean(data['final_lead_times_minutes'])
                comparison_data.append({
                    'Issue Type': issue_type.upper(),
                    'Reply Type': 'Final Reply',
                    'Average Lead Time (minutes)': final_avg,
                    'Formatted Time': format_lead_time(final_avg)
                })
            except:
                pass
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        
        # Pilih unit yang paling appropriate untuk chart
        max_lead_time = df_comparison['Average Lead Time (minutes)'].max()
        if max_lead_time > 1440:  # > 1 day
            df_comparison['Average Lead Time'] = df_comparison['Average Lead Time (minutes)'] / 1440
            y_title = 'Average Lead Time (days)'
        elif max_lead_time > 60:  # > 1 hour
            df_comparison['Average Lead Time'] = df_comparison['Average Lead Time (minutes)'] / 60
            y_title = 'Average Lead Time (hours)'
        else:
            df_comparison['Average Lead Time'] = df_comparison['Average Lead Time (minutes)']
            y_title = 'Average Lead Time (minutes)'
        
        fig_comparison = px.bar(
            df_comparison, 
            x='Issue Type', 
            y='Average Lead Time',
            color='Reply Type',
            title=f'Average Lead Time Comparison by Issue Type',
            labels={'Average Lead Time': y_title, 'Issue Type': 'Issue Type'},
            barmode='group',
            color_discrete_map={
                'First Reply': '#2E86AB',
                'Final Reply': '#A23B72'
            }
        )
        
        # Format tooltip untuk menunjukkan formatted time
        fig_comparison.update_traces(
            hovertemplate='<b>%{x}</b><br>%{data.name}<br>Average: %{customdata}<extra></extra>',
            customdata=df_comparison['Formatted Time']
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
    else:
        st.info("No data available for comparison chart")
                    
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
    
    # Raw Data Access (if available)
    if st.checkbox("Show Raw Analysis Data"):
        st.markdown("#### 🔧 Raw Analysis Data")
        
        if '_raw_qa_pairs' in result:
            with st.expander("Q-A Pairs Raw Data"):
                qa_data = []
                for i, pair in enumerate(result['_raw_qa_pairs']):
                    qa_data.append({
                        'Pair': i + 1,
                        'Question': pair.get('question', '')[:100] + '...',
                        'Answered': '✅' if pair.get('is_answered') else '❌',
                        'Answer': pair.get('answer', '')[:100] + '...' if pair.get('answer') else 'No Answer',
                        'Lead Time (min)': pair.get('lead_time_minutes', 'N/A')
                    })
                st.dataframe(pd.DataFrame(qa_data))
        
        if '_raw_reply_analysis' in result:
            with st.expander("Reply Analysis Details"):
                reply_analysis = result['_raw_reply_analysis']
                
                # PERBAIKAN: Tampilkan dalam format yang lebih readable
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Basic Info**")
                    st.write(f"**Issue Type:** {reply_analysis.get('issue_type', 'N/A')}")
                    st.write(f"**Customer Leave:** {reply_analysis.get('customer_leave', False)}")
                    st.write(f"**Requirement Compliant:** {reply_analysis.get('requirement_compliant', False)}")
                
                with col2:
                    st.markdown("**Reply Status**")
                    st.write(f"**First Reply Found:** {reply_analysis.get('first_reply') is not None}")
                    st.write(f"**Final Reply Found:** {reply_analysis.get('final_reply') is not None}")
                
                # First Reply Details
                if reply_analysis.get('first_reply'):
                    st.markdown("**First Reply Details**")
                    first_reply = reply_analysis['first_reply']
                    st.write(f"Message: {first_reply.get('message', '')[:200]}...")
                    st.write(f"Lead Time: {first_reply.get('lead_time_minutes', 'N/A')} min")
                    st.write(f"Note: {first_reply.get('note', 'N/A')}")
                
                # Final Reply Details
                if reply_analysis.get('final_reply'):
                    st.markdown("**Final Reply Details**")
                    final_reply = reply_analysis['final_reply']
                    st.write(f"Message: {final_reply.get('message', '')[:200]}...")
                    
                    if final_reply.get('lead_time_days'):
                        st.write(f"Lead Time: {final_reply.get('lead_time_days')} days")
                    else:
                        st.write(f"Lead Time: {final_reply.get('lead_time_minutes', 'N/A')} min")
                    
                    st.write(f"Note: {final_reply.get('note', 'N/A')}")
                
                # Tampilkan JSON mentah hanya jika diminta
                if st.checkbox("Show Raw JSON"):
                    st.json(reply_analysis)
                
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
    st.markdown("## 🚨 Special Cases Analysis")
    
    successful = [r for r in results if r['status'] == 'success']
    
    # Extract special cases
    customer_leave_cases = [r for r in successful if r.get('customer_leave')]
    
    # SUMMARY CARDS
    st.markdown("### 📊 Special Cases Summary")
    st.metric("Customer Leave Cases", len(customer_leave_cases))
    
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

def display_debug_tab(results, stats):
    """Display debug information"""
    st.markdown("## 🐛 Debug Information")
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Analysis Status")
        st.metric("Successful Analyses", len(successful))
        st.metric("Failed Analyses", len(failed))
        st.metric("Success Rate", f"{(len(successful)/len(results))*100:.1f}%" if results else "0%")
    
    with col2:
        st.markdown("### ⚠️ Common Issues")
        
        # Check for negative lead times
        negative_lead_times = []
        for result in successful:
            first_lt = result.get('first_reply_lead_time_minutes')
            final_lt = result.get('final_reply_lead_time_minutes')
            
            if (first_lt is not None and first_lt < 0) or (final_lt is not None and final_lt < 0):
                negative_lead_times.append(result['ticket_id'])
        
        if negative_lead_times:
            st.error(f"❌ {len(negative_lead_times)} tickets with negative lead times")
        else:
            st.success("✅ No negative lead times detected")
        
        # Check for missing required replies
        missing_required = []
        for result in successful:
            issue_type = result['final_issue_type']
            if issue_type in ['serious', 'complaint'] and not result['first_reply_found']:
                missing_required.append(result['ticket_id'])
            elif issue_type == 'normal' and not result['final_reply_found'] and not result.get('customer_leave'):
                missing_required.append(result['ticket_id'])
        
        if missing_required:
            st.warning(f"⚠️ {len(missing_required)} tickets missing required replies")
        else:
            st.success("✅ All required replies present")
    
    # Show failed analyses
    if failed:
        st.markdown("### ❌ Failed Analyses")
        for result in failed[:5]:
            with st.expander(f"Failed: {result['ticket_id']}"):
                st.write(f"Reason: {result['failure_reason']}")
    
    # Raw data inspection
    st.markdown("### 🔍 Raw Data Inspection")
    if successful:
        sample_ticket = successful[0]
        with st.expander("Sample Ticket Data Structure"):
            st.json({k: v for k, v in sample_ticket.items() if not k.startswith('_raw')})
    
    # System information
    st.markdown("### 🔧 System Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Python Version", sys.version.split()[0])
    with col2:
        st.metric("Pandas Version", pd.__version__)
    with col3:
        st.metric("Analysis Time", f"{stats.get('analysis_duration_seconds', 0):.1f}s" if stats else "N/A")
        
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




























