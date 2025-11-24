# app.py - FIXED Streamlit Dashboard dengan REAL Analysis
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
    # Import semua class yang diperlukan
    from Chat_Analyzer_System import (
        DataPreprocessor, CompleteAnalysisPipeline, 
        ResultsExporter, ModelTrainer, Config
    )
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Analysis modules not available: {e}")
    ANALYSIS_AVAILABLE = False

st.set_page_config(
    page_title="Live Chat Analysis Dashboard",
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

    .metric-card .trend {
        font-size: 0.75rem;
        margin-top: 8px;
        padding: 4px 8px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.2);
        display: inline-block;
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
</style>
""", unsafe_allow_html=True)

def main_interface():
    """Main interface dengan upload dan analysis options"""
    st.markdown('<h1 class="main-header">🤖 Live Chat Lead Time Analysis</h1>', unsafe_allow_html=True)
    st.markdown("Upload your chat data Excel file for Live Chat performance analysis")
    
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
        
        uploaded_file = st.file_uploader(
            "Upload Excel File", 
            type=['xlsx', 'xls'],
            help="Format kolom: No, Ticket Number, Role, Sender, Message Date, Message"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Preview data
            if st.checkbox("Preview uploaded data"):
                try:
                    df_preview = pd.read_excel(uploaded_file)
                    st.write(f"Data shape: {df_preview.shape}")
                    st.dataframe(df_preview.head(10))
                    
                    # Check required columns
                    required_cols = ['Ticket Number', 'Role', 'Sender', 'Message Date', 'Message']
                    missing_cols = [col for col in required_cols if col not in df_preview.columns]
                    if missing_cols:
                        st.error(f"❌ Missing required columns: {missing_cols}")
                    else:
                        st.success("✅ All required columns present")
                        
                except Exception as e:
                    st.error(f"Error previewing data: {e}")
        
        st.markdown("---")
        st.header("⚙️ Analysis Settings")
        
        max_tickets = st.slider(
            "Maximum Tickets to Analyze",
            min_value=10,
            max_value=100000,
            value=100,
            help="Limit number of tickets for faster analysis"
        )
        
        analysis_type = st.radio(
            "Analysis Type",
            ["Quick Analysis", "Comprehensive Analysis"],
            help="Quick: Basic metrics, Comprehensive: Full analysis dengan ML"
        )
        
        st.markdown("---")
        
        # Analysis Button di Sidebar
        if uploaded_file is not None:
            if st.button("🚀 START ANALYSIS", type="primary", use_container_width=True):
                with st.spinner("🔄 Starting analysis..."):
                    results, stats, excel_path = run_analysis(uploaded_file, max_tickets, analysis_type)
                    
                    if results is not None and stats is not None:
                        st.session_state.analysis_complete = True
                        st.session_state.analysis_results = results
                        st.session_state.analysis_stats = stats
                        st.session_state.excel_file_path = excel_path
                        st.rerun()
                    else:
                        st.error("❌ Analysis failed. Please check your data format.")
        
        st.markdown("---")
        st.markdown("### 📖 How to Use")
        st.info("""
        1. Upload an Excel file containing the chat data
        2. Configure the analysis settings  
        3. Click "Start Analysis"
        4. View the results & download the reports
        
        **Required columns:**
        - Ticket Number
        - Role  
        - Sender
        - Message Date
        - Message
        """)
    
    # Main content area
    if uploaded_file is not None:
        st.markdown("---")
        st.markdown("## 🚀 Ready for Analysis")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🎯 Start Analysis", type="primary", use_container_width=True):
                with st.spinner("🔄 Starting analysis..."):
                    results, stats, excel_path = run_analysis(uploaded_file, max_tickets, analysis_type)
                    
                    if results is not None and stats is not None:
                        st.session_state.analysis_complete = True
                        st.session_state.analysis_results = results
                        st.session_state.analysis_stats = stats
                        st.session_state.excel_file_path = excel_path
                        st.rerun()
                    else:
                        st.error("❌ Analysis failed. Please check your data format.")
    
    else:
        # Show sample data option - FIXED VERSION
        st.markdown("---")
        st.markdown("## 🎯 Try with Sample Data")
        
        sample_file_path = "data/raw_conversation.xlsx"
        if os.path.exists(sample_file_path):
            if st.button("🧪 Analyze Sample Data", use_container_width=True):
                with st.spinner("Loading sample data..."):
                    try:
                        # Use actual file path instead of MockFile
                        results, stats, excel_path = run_analysis(sample_file_path, 50, "Quick Analysis")
                        
                        if results is not None and stats is not None:
                            st.session_state.analysis_complete = True
                            st.session_state.analysis_results = results
                            st.session_state.analysis_stats = stats
                            st.session_state.excel_file_path = excel_path
                            st.rerun()
                        else:
                            st.error("Failed to analyze sample data")
                    except Exception as e:
                        st.error(f"Error analyzing sample data: {e}")
                        st.error(traceback.format_exc())
        else:
            st.info("📁 No sample data found. Please upload your own Excel file.")

def run_direct_analysis(df, max_tickets, analysis_type):
    """Run analysis langsung dari DataFrame - ROBUST VERSION"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Initializing analysis pipeline...")
        progress_bar.progress(10)
        
        # 1. PREPROCESS DATA - ROBUST APPROACH
        status_text.text("📊 Preprocessing data...")
        progress_bar.progress(30)
        
        # Coba beberapa approach untuk preprocessing
        processed_data = None
        
        # Approach 1: Langsung gunakan DataFrame yang sudah clean
        try:
            preprocessor = DataPreprocessor()
            # Coba method clean_data
            if hasattr(preprocessor, 'clean_data'):
                processed_data = preprocessor.clean_data(df)
                st.info("✅ Used clean_data method for preprocessing")
            # Coba method load_raw_data (jika ada)
            elif hasattr(preprocessor, 'load_raw_data'):
                # Simpan ke file temporary dulu
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
                    df.to_excel(tmp_file.name, index=False)
                    processed_data = preprocessor.load_raw_data(tmp_file.name)
                st.info("✅ Used load_raw_data method for preprocessing")
            else:
                # Fallback: langsung gunakan DataFrame asli
                processed_data = df
                st.info("✅ Using original DataFrame (no preprocessing needed)")
        except Exception as e:
            st.warning(f"⚠️ Preprocessing failed: {e}. Using original DataFrame.")
            processed_data = df
        
        if processed_data is None or len(processed_data) == 0:
            st.error("❌ No data available for analysis")
            return None, None, None
            
        st.success(f"✅ Ready to analyze {len(processed_data)} rows")
        
        # 2. RUN ANALYSIS PIPELINE
        status_text.text("🔍 Analyzing conversations with AI...")
        progress_bar.progress(60)
        
        pipeline = CompleteAnalysisPipeline()
        results, stats = pipeline.analyze_all_tickets(processed_data, max_tickets=max_tickets)
        
        if not results:
            st.error("❌ No results generated from analysis")
            return None, None, None
            
        successful = [r for r in results if r['status'] == 'success']
        st.success(f"✅ Successfully analyzed {len(successful)} conversations")
        
        # 3. EXPORT RESULTS
        status_text.text("💾 Generating comprehensive report...")
        progress_bar.progress(80)
        
        exporter = ResultsExporter()
        excel_path = exporter.export_comprehensive_results(results, stats)
        
        # 4. CREATE VISUALIZATIONS
        status_text.text("📊 Creating visualizations...")
        progress_bar.progress(95)
        
        try:
            exporter.create_comprehensive_visualizations(results, stats)
        except Exception as e:
            st.warning(f"Visualizations skipped: {e}")
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return results, stats, excel_path
        
    except Exception as e:
        st.error(f"❌ Analysis error: {str(e)}")
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None, None, None

def run_analysis(uploaded_file, max_tickets, analysis_type):
    """Run analysis pada uploaded file - FIXED VERSION"""
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Loading uploaded file...")
        progress_bar.progress(20)
        
        # 1. LOAD DATA DARI UPLOADED FILE
        if hasattr(uploaded_file, 'read'):
            # Untuk file yang diupload via Streamlit
            df = pd.read_excel(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows from uploaded file")
        elif isinstance(uploaded_file, str) and os.path.exists(uploaded_file):
            # Untuk file path (sample data)
            df = pd.read_excel(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows from sample file")
        else:
            st.error("❌ Invalid file type or path")
            return None, None, None
        
        # 2. CHECK REQUIRED COLUMNS
        required_cols = ['Ticket Number', 'Role', 'Sender', 'Message Date', 'Message']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            return None, None, None
        
        # 3. RUN ANALYSIS
        return run_direct_analysis(df, max_tickets, analysis_type)
        
    except Exception as e:
        st.error(f"❌ File loading error: {str(e)}")
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None, None, None
        
def display_complete_results():
    """Display COMPLETE analysis results dengan semua tab dan download"""
    
    results = st.session_state.analysis_results
    stats = st.session_state.analysis_stats
    
    # Validasi stats
    if not stats:
        st.error("❌ No analysis statistics available")
        return
    
    st.markdown("---")
    st.markdown('<h1 class="main-header">📊 Analysis Results</h1>', unsafe_allow_html=True)
    
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
            total_issues = sum(stats['issue_type_distribution'].values())
            metric_value = total_issues
        else:
            metric_value = "N/A"
        
        st.markdown("""
        <div class="metric-card">
            <h3>Issues Found</h3>
            <h1>{}</h1>
        </div>
        """.format(metric_value), unsafe_allow_html=True)

    # SPECIAL CASES SUMMARY
    if 'reply_effectiveness' in stats:
        eff = stats['reply_effectiveness']
        if eff.get('customer_leave_cases', 0) > 0 or eff.get('follow_up_cases', 0) > 0:
            st.markdown("---")
            st.markdown("## 🚨 Special Cases Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="special-case">', unsafe_allow_html=True)
                st.metric("Customer Leave Cases", eff.get('customer_leave_cases', 0))
                st.caption("Conversations where customer left without response")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="special-case">', unsafe_allow_html=True)
                st.metric("Follow-up Cases", eff.get('follow_up_cases', 0))
                st.caption("Issues resolved in different tickets")
                st.markdown('</div>', unsafe_allow_html=True)

    # DOWNLOAD SECTION
    st.markdown("---")
    st.markdown("## 💾 Download Complete Analysis Results")
    
    if st.session_state.get('excel_file_path') and os.path.exists(st.session_state.excel_file_path):
        with open(st.session_state.excel_file_path, "rb") as f:
            excel_data = f.read()
        
        st.download_button(
            label="📥 DOWNLOAD COMPLETE EXCEL REPORT",
            data=excel_data,
            file_name=f"chat_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True
        )
        st.success("✅ Excel report contains ALL parsed data: Q-A pairs, main issues, reply analysis, timestamps, lead time summary, and detailed metrics!")
        
        # Show file info
        file_size = os.path.getsize(st.session_state.excel_file_path) / (1024 * 1024)  # MB
        st.info(f"📊 Report includes: Detailed analysis sheets with raw data ({file_size:.1f} MB)")
    else:
        st.error("❌ Excel file not available for download. Analysis may not have exported properly.")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # TABS
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Overview", "🎯 Main Issues", "⏱️ Lead Times", "📊 Performance", "💬 Special Cases", "📋 All Data", "🐛 Debug"
    ])
    
    with tab1:
        display_overview_tab(results, stats)
    
    with tab2:
        display_main_issues_tab(results)
    
    with tab3:
        display_lead_time_tab(results, stats)
    
    with tab4:
        display_performance_tab(results, stats)
    
    with tab5:
        display_special_cases_tab(results, stats)
    
    with tab6:
        display_raw_data_tab(results)

    with tab7:
        display_debug_tab(results, stats)

    # NEW ANALYSIS BUTTON
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Analyze New Data", type="secondary", use_container_width=True):
            # Reset session state
            for key in ['analysis_complete', 'analysis_results', 'analysis_stats', 'excel_file_path']:
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
                    'excellent': '#2E86AB',
                    'good': '#A23B72', 
                    'fair': '#F18F01',
                    'poor': '#C73E1D'
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
            st.metric("Count of First Reply", lt_stats['first_reply_samples'])
        with col4:
            st.metric("Count of Final Reply", lt_stats['final_reply_samples'])
    
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
            st.metric("Both Replies Found", f"{eff['both_replies_found_rate']*100:.1f}%")

def display_main_issues_tab(results):
    """Display main issues dengan first/final reply details"""
    st.markdown("## 🎯 Main Issues Analysis")
    
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
            both_replies_found = sum(1 for r in successful if r['first_reply_found'] and r['final_reply_found'])
            st.metric("Both Replies Found", f"{both_replies_found}/{len(successful)}")
        
        with col4:
            avg_quality = np.mean([r['quality_score'] for r in successful])
            st.metric("Avg Quality Score", f"{avg_quality:.1f}/6")

        # Main issues table
        st.markdown("### 📋 All Main Issues")
        display_data = []
        for result in successful:
            special_notes = []
            if result.get('customer_leave'):
                special_notes.append("🚶 Customer Leave")
            if result.get('follow_up_ticket'):
                special_notes.append("🔄 Follow-up")
            
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
    st.markdown(f'<div class="message-box"><strong>Question:</strong> {result["main_question"]}</div>', unsafe_allow_html=True)
    st.caption(f"Detected as: {result['final_issue_type'].upper()} (Confidence: {result['detection_confidence']:.2f})")
    
    # SPECIAL CASES SECTION
    special_cases = result.get('_raw_reply_analysis', {}).get('special_cases', [])
    if special_cases:
        st.markdown("### 🚨 Special Conditions")
        
        for case in special_cases:
            if case == 'customer_leave':
                st.markdown(f'<div class="special-case">🚶 **Customer Leave**: Customer left conversation without response</div>', unsafe_allow_html=True)
            elif case == 'escalation_reply':
                st.markdown(f'<div class="special-case">🔄 **Escalation Required**: Issue needs follow-up from other team</div>', unsafe_allow_html=True)
            elif case == 'first_as_final':
                st.markdown(f'<div class="special-case">🔀 **First as Final**: Using first reply as final reply</div>', unsafe_allow_html=True)
            elif case == 'customer_leave_final':
                st.markdown(f'<div class="special-case">🚶 **Customer Leave Final**: Used last operator message due to customer leave</div>', unsafe_allow_html=True)
            elif case == 'fallback_reply':
                st.markdown(f'<div class="special-case">🔄 **Fallback Reply**: Used fallback reply method</div>', unsafe_allow_html=True)

    # Additional special notes
    special_notes = []
    if result.get('customer_leave'):
        special_notes.append("🚶 **Customer Leave**: Customer left conversation without response")
    if result.get('follow_up_ticket'):
        special_notes.append(f"🔄 **Follow-up**: Resolved in ticket {result['follow_up_ticket']}")
    if result.get('customer_leave_note'):
        special_notes.append(f"📝 **Note**: {result['customer_leave_note']}")
    
    if special_notes:
        st.markdown("### 📋 Additional Notes")
        for note in special_notes:
            st.markdown(f'<div class="special-case">{note}</div>', unsafe_allow_html=True)
    
    # First Reply Section
    st.markdown("#### 🔄 First Reply Analysis")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if result['first_reply_found']:
            first_reply_msg = result.get('first_reply_message', 'No message content available')
            st.markdown(f'<div class="message-box"><strong>First Reply:</strong> {first_reply_msg}</div>', unsafe_allow_html=True)
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
            reply_type = result.get('_raw_reply_analysis', {}).get('final_reply', {}).get('reply_type', 'standard')
            
            # Add reply type badge
            type_badge = ""
            if 'escalation' in reply_type:
                type_badge = "🔄 "
            elif 'customer_leave' in reply_type:
                type_badge = "🚶 "
            elif 'first_as_final' in reply_type:
                type_badge = "🔀 "
            
            st.markdown(f'<div class="message-box"><strong>Final Reply ({type_badge}{reply_type.replace("_", " ").title()}):</strong> {final_reply_msg}</div>', unsafe_allow_html=True)
        else:
            if result['final_issue_type'] == 'normal' and not result.get('customer_leave'):
                st.error("❌ No final reply found - REQUIRED for normal issues")
            else:
                st.info("ℹ️ No final reply found - May be handled in follow-up")
    
    with col2:
        if result['final_reply_found']:
            st.metric("Lead Time", f"{result.get('final_reply_lead_time_minutes', 'N/A')} min")
            st.metric("Time Format", result.get('final_reply_lead_time_hhmmss', 'N/A'))
        else:
            st.metric("Status", "Not Found")
    
    # Performance Metrics - FIXED VERSION
    st.markdown("#### 📊 Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        performance_color = {
            'excellent': 'green',
            'good': 'blue', 
            'fair': 'orange',
            'poor': 'red'
        }.get(result['performance_rating'], 'gray')
        
        # FIX: Gunakan delta="normal" untuk menghindari error delta_color
        st.metric(
            "Performance Rating", 
            result['performance_rating'].upper(),
            delta="normal",  # Tambahkan delta value
            delta_color=performance_color
        )
    
    with col2:
        quality_color = "green" if result['quality_score'] >= 4 else "orange" if result['quality_score'] >= 2 else "red"
        st.metric(
            "Quality Score", 
            f"{result['quality_score']}/6",
            delta="normal",  # Tambahkan delta value
            delta_color=quality_color
        )
    
    with col3:
        st.metric("Total Messages", result['total_messages'])
    
    with col4:
        answer_rate = (result['answered_pairs'] / result['total_qa_pairs']) * 100 if result['total_qa_pairs'] > 0 else 0
        st.metric("Answer Rate", f"{answer_rate:.1f}%")
    
    # Conversation Statistics
    st.markdown("#### 📈 Conversation Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Q-A Pairs", result['total_qa_pairs'])
    
    with col2:
        st.metric("Answered Pairs", result['answered_pairs'])
    
    with col3:
        st.metric("Unanswered Pairs", result['total_qa_pairs'] - result['answered_pairs'])
    
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
        
        if '_raw_main_issue' in result:
            with st.expander("Main Issue Detection Details"):
                main_issue = result['_raw_main_issue']
                st.json(main_issue)
        
        if '_raw_reply_analysis' in result:
            with st.expander("Reply Analysis Details"):
                reply_analysis = result['_raw_reply_analysis']
                st.json(reply_analysis)

def display_lead_time_tab(results, stats):
    """Display lead time analysis dengan breakdown per issue type"""
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
            st.metric("First Reply Median", f"{overall_lt['first_reply_median_minutes']:.1f} min")
        with col4:
            st.metric("Samples", f"{overall_lt['first_reply_samples']} / {overall_lt['final_reply_samples']}")
    
    # Lead Time by Issue Type
    if 'issue_type_lead_times' in stats:
        st.markdown("### 📈 Lead Time by Issue Type")
        
        # Define the desired order
        desired_order = ['normal', 'serious', 'complaint']
        
        issue_type_data = []
        for issue_type in desired_order:
            if issue_type in stats['issue_type_lead_times']:
                lt_stats = stats['issue_type_lead_times'][issue_type]
                first_avg = lt_stats.get('first_reply_avg_minutes')
                final_avg = lt_stats.get('final_reply_avg_minutes')
                
                if first_avg is not None or final_avg is not None:
                    issue_type_data.append({
                        'Issue Type': issue_type.upper(),
                        'First Reply Avg (min)': f"{first_avg:.1f}" if first_avg is not None else "N/A",
                        'Final Reply Avg (min)': f"{final_avg:.1f}" if final_avg is not None else "N/A",
                        'First Reply Samples': lt_stats['first_reply_samples'],
                        'Final Reply Samples': lt_stats['final_reply_samples']
                    })
        
        if issue_type_data:
            df_issue_lt = pd.DataFrame(issue_type_data)
            st.dataframe(df_issue_lt, use_container_width=True)
            
            # Visualization
            st.markdown("### 📊 Lead Time Comparison by Issue Type")
            
            # Prepare data for visualization
            viz_data = []
            for issue_type in desired_order:
                if issue_type in stats['issue_type_lead_times']:
                    lt_stats = stats['issue_type_lead_times'][issue_type]
                    first_avg = lt_stats.get('first_reply_avg_minutes')
                    final_avg = lt_stats.get('final_reply_avg_minutes')
                    
                    if first_avg is not None:
                        viz_data.append({
                            'Issue Type': issue_type.upper(),
                            'Lead Time Type': 'First Reply',
                            'Average Lead Time (min)': first_avg,
                            'Samples': lt_stats['first_reply_samples']
                        })
                    if final_avg is not None:
                        viz_data.append({
                            'Issue Type': issue_type.upper(),
                            'Lead Time Type': 'Final Reply',
                            'Average Lead Time (min)': final_avg,
                            'Samples': lt_stats['final_reply_samples']
                        })
            
            if viz_data:
                df_viz = pd.DataFrame(viz_data)
                
                fig = px.bar(
                    df_viz, 
                    x='Issue Type', 
                    y='Average Lead Time (min)', 
                    color='Lead Time Type',
                    title='Average Lead Time by Issue Type and Reply Type',
                    barmode='group',
                    labels={'Average Lead Time (min)': 'Average Lead Time (minutes)'},
                    color_discrete_map={
                        'First Reply': '#2E86AB',
                        'Final Reply': '#A23B72'
                    },
                    category_orders={"Issue Type": [t.upper() for t in desired_order]}
                )
                
                # Add sample size annotations
                for i, row in df_viz.iterrows():
                    fig.add_annotation(
                        x=row['Issue Type'],
                        y=row['Average Lead Time (min)'],
                        text=f"n={row['Samples']}",
                        showarrow=False,
                        yshift=10,
                        font=dict(size=10)
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
            # Box plot
            fig_box = px.box(
                x=lead_times,
                title='Final Reply Lead Time Distribution',
                labels={'x': 'Lead Time (minutes)'},
                color_discrete_sequence=['#A23B72']
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Statistics
        st.markdown("#### 📈 Lead Time Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{np.mean(lead_times):.1f} min")
        with col2:
            st.metric("Median", f"{np.median(lead_times):.1f} min")
        with col3:
            st.metric("Std Dev", f"{np.std(lead_times):.1f} min")
        with col4:
            st.metric("Range", f"{np.min(lead_times):.1f} - {np.max(lead_times):.1f} min")
    else:
        st.info("No lead time data available for detailed analysis")

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
            if result.get('follow_up_ticket'):
                special_notes.append("Follow-up")
            
            perf_metrics.append({
                'Ticket ID': result['ticket_id'],
                'Issue Type': result['final_issue_type'],
                'Performance': result['performance_rating'].upper(),
                'Quality Score': result['quality_score'],
                'First Reply LT': result.get('first_reply_lead_time_minutes', 'N/A'),
                'Final Reply LT': result.get('final_reply_lead_time_minutes', 'N/A'),
                'First Reply': '✅' if result['first_reply_found'] else '❌',
                'Final Reply': '✅' if result['final_reply_found'] else '❌',
                'Special Notes': ', '.join(special_notes) if special_notes else '-'
            })
        
        df_perf_metrics = pd.DataFrame(perf_metrics)
        st.dataframe(df_perf_metrics, use_container_width=True)

def display_special_cases_tab(results, stats):
    """Display special cases analysis (customer leave & follow-up & escalation)"""
    st.markdown("## 🚨 Special Cases Analysis")
    
    successful = [r for r in results if r['status'] == 'success']
    
    # Extract all special cases
    customer_leave_cases = [r for r in successful if r.get('customer_leave')]
    escalation_cases = [r for r in successful if r.get('_raw_reply_analysis', {}).get('special_cases', []) and 
                       any('escalation' in case for case in r['_raw_reply_analysis']['special_cases'])]
    first_as_final_cases = [r for r in successful if r.get('_raw_reply_analysis', {}).get('special_cases', []) and 
                           any('first_as_final' in case for case in r['_raw_reply_analysis']['special_cases'])]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🚶 Customer Leave Cases")
        if customer_leave_cases:
            st.info(f"Found {len(customer_leave_cases)} conversations where customer left")
            
            leave_data = []
            for result in customer_leave_cases:
                # Cari final reply message yang digunakan
                final_reply_msg = "No final reply used"
                if result.get('final_reply_message'):
                    final_reply_msg = result['final_reply_message']
                elif result.get('first_reply_message') and result.get('_raw_reply_analysis', {}).get('special_cases', []):
                    if any('first_as_final' in case for case in result['_raw_reply_analysis']['special_cases']):
                        final_reply_msg = f"FIRST AS FINAL: {result['first_reply_message']}"
                
                leave_data.append({
                    'Ticket ID': result['ticket_id'],
                    'Issue Type': result['final_issue_type'],
                    'Main Question': result['main_question'][:60] + '...',
                    'Final Reply Used': final_reply_msg[:80] + '...' if len(final_reply_msg) > 80 else final_reply_msg,
                    'Performance': result['performance_rating'].upper()
                })
            
            df_leave = pd.DataFrame(leave_data)
            st.dataframe(df_leave, use_container_width=True)
            
            # Show impact analysis
            st.markdown("**Impact Analysis:**")
            col1, col2 = st.columns(2)
            with col1:
                perf_counts = pd.Series([r['performance_rating'] for r in customer_leave_cases]).value_counts()
                if not perf_counts.empty:
                    st.metric("Most Common Performance", perf_counts.index[0].upper())
            with col2:
                avg_quality = np.mean([r['quality_score'] for r in customer_leave_cases])
                st.metric("Avg Quality Score", f"{avg_quality:.1f}/6")
        else:
            st.success("✅ No customer leave cases detected")
    
    with col2:
        st.markdown("### 🔄 Escalation Cases")
        if escalation_cases:
            st.info(f"Found {len(escalation_cases)} issues requiring follow-up/escalation")
            
            escalation_data = []
            for result in escalation_cases:
                escalation_data.append({
                    'Ticket ID': result['ticket_id'],
                    'Issue Type': result['final_issue_type'],
                    'Main Question': result['main_question'][:60] + '...',
                    'Escalation Message': result.get('final_reply_message', 'No message')[:100] + '...' if len(result.get('final_reply_message', '')) > 100 else result.get('final_reply_message', 'No message'),
                    'First Reply Found': '✅' if result['first_reply_found'] else '❌',
                    'Performance': result['performance_rating'].upper()
                })
            
            df_escalation = pd.DataFrame(escalation_data)
            st.dataframe(df_escalation, use_container_width=True)
            
            # Show escalation analysis
            st.markdown("**Escalation Analysis:**")
            issue_counts = pd.Series([r['final_issue_type'] for r in escalation_cases]).value_counts()
            if not issue_counts.empty:
                st.write("**Issue Types:**")
                for issue_type, count in issue_counts.items():
                    st.write(f"- {issue_type.upper()}: {count} cases")
        else:
            st.success("✅ No escalation cases detected")
    
    with col3:
        st.markdown("### 🔀 First as Final Cases")
        if first_as_final_cases:
            st.info(f"Found {len(first_as_final_cases)} cases using first reply as final")
            
            first_final_data = []
            for result in first_as_final_cases:
                first_final_data.append({
                    'Ticket ID': result['ticket_id'],
                    'Issue Type': result['final_issue_type'],
                    'Main Question': result['main_question'][:60] + '...',
                    'First/Final Message': result.get('first_reply_message', 'No message')[:80] + '...',
                    'Customer Leave': '✅' if result.get('customer_leave') else '❌',
                    'Performance': result['performance_rating'].upper()
                })
            
            df_first_final = pd.DataFrame(first_final_data)
            st.dataframe(df_first_final, use_container_width=True)
            
            # Show analysis
            st.markdown("**Analysis:**")
            col1, col2 = st.columns(2)
            with col1:
                customer_leave_count = sum(1 for r in first_as_final_cases if r.get('customer_leave'))
                st.metric("With Customer Leave", customer_leave_count)
            with col2:
                serious_complaint_count = sum(1 for r in first_as_final_cases if r['final_issue_type'] in ['serious', 'complaint'])
                st.metric("Serious/Complaint Issues", serious_complaint_count)
        else:
            st.success("✅ No first-as-final cases detected")

    # DETAILED SPECIAL CASES BREAKDOWN
    if customer_leave_cases or escalation_cases or first_as_final_cases:
        st.markdown("---")
        st.markdown("### 📊 Detailed Special Cases Breakdown")
        
        # Create summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_special = len(customer_leave_cases) + len(escalation_cases) + len(first_as_final_cases)
            st.metric("Total Special Cases", total_special)
        
        with col2:
            unique_tickets = len(set([r['ticket_id'] for r in customer_leave_cases + escalation_cases + first_as_final_cases]))
            st.metric("Unique Tickets", unique_tickets)
        
        with col3:
            if successful:
                special_rate = (total_special / len(successful)) * 100
                st.metric("Special Cases Rate", f"{special_rate:.1f}%")
        
        with col4:
            avg_quality_special = np.mean([r['quality_score'] for r in customer_leave_cases + escalation_cases + first_as_final_cases])
            st.metric("Avg Quality (Special)", f"{avg_quality_special:.1f}/6")

        # Performance comparison
        st.markdown("#### 📈 Performance Comparison: Special vs Normal Cases")
        
        special_cases = customer_leave_cases + escalation_cases + first_as_final_cases
        normal_cases = [r for r in successful if r not in special_cases]
        
        if special_cases and normal_cases:
            col1, col2 = st.columns(2)
            
            with col1:
                # Performance distribution for special cases
                special_perf = pd.Series([r['performance_rating'] for r in special_cases]).value_counts()
                fig_special = px.pie(
                    values=special_perf.values,
                    names=special_perf.index,
                    title='Performance - Special Cases',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_special, use_container_width=True)
            
            with col2:
                # Performance distribution for normal cases
                normal_perf = pd.Series([r['performance_rating'] for r in normal_cases]).value_counts()
                fig_normal = px.pie(
                    values=normal_perf.values,
                    names=normal_perf.index,
                    title='Performance - Normal Cases',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_normal, use_container_width=True)
            
            # Quality score comparison
            st.markdown("#### 🎯 Quality Score Comparison")
            comparison_data = []
            for result in special_cases:
                comparison_data.append({
                    'Category': 'Special Cases',
                    'Quality Score': result['quality_score'],
                    'Issue Type': result['final_issue_type']
                })
            for result in normal_cases:
                comparison_data.append({
                    'Category': 'Normal Cases', 
                    'Quality Score': result['quality_score'],
                    'Issue Type': result['final_issue_type']
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            fig_quality = px.box(
                df_comparison, 
                x='Category', 
                y='Quality Score',
                title='Quality Score Distribution: Special vs Normal Cases',
                color='Category',
                color_discrete_map={
                    'Special Cases': '#FF6B6B',
                    'Normal Cases': '#4ECDC4'
                }
            )
            st.plotly_chart(fig_quality, use_container_width=True)

        # RECOMMENDATIONS SECTION
        st.markdown("---")
        st.markdown("### 💡 Recommendations for Special Cases")
        
        recommendations = []
        
        if customer_leave_cases:
            recommendations.append({
                'Issue': 'Customer Leave Cases',
                'Recommendation': 'Implement proactive engagement strategies to prevent early drop-offs',
                'Priority': 'High',
                'Impact': f"{len(customer_leave_cases)} cases ({len(customer_leave_cases)/len(successful)*100:.1f}%)"
            })
        
        if escalation_cases:
            recommendations.append({
                'Issue': 'Escalation Cases', 
                'Recommendation': 'Improve first-contact resolution and empower agents with better solutions',
                'Priority': 'Medium',
                'Impact': f"{len(escalation_cases)} cases ({len(escalation_cases)/len(successful)*100:.1f}%)"
            })
        
        if first_as_final_cases:
            recommendations.append({
                'Issue': 'First-as-Final Cases',
                'Recommendation': 'Ensure proper follow-up procedures for cases where initial reply is insufficient',
                'Priority': 'Medium', 
                'Impact': f"{len(first_as_final_cases)} cases ({len(first_as_final_cases)/len(successful)*100:.1f}%)"
            })
        
        if recommendations:
            for rec in recommendations:
                with st.expander(f"🚨 {rec['Issue']} - Priority: {rec['Priority']}"):
                    st.write(f"**Recommendation:** {rec['Recommendation']}")
                    st.write(f"**Impact:** {rec['Impact']}")
                    
                    # Show sample tickets
                    if rec['Issue'] == 'Customer Leave Cases':
                        sample_tickets = [r['ticket_id'] for r in customer_leave_cases[:3]]
                    elif rec['Issue'] == 'Escalation Cases':
                        sample_tickets = [r['ticket_id'] for r in escalation_cases[:3]]
                    else:
                        sample_tickets = [r['ticket_id'] for r in first_as_final_cases[:3]]
                    
                    st.write(f"**Sample Tickets:** {', '.join(sample_tickets)}")
    else:
        st.success("🎉 No special cases detected in this analysis!")

def display_raw_data_tab(results):
    """Display raw data tab untuk melihat semua hasil parse"""
    st.markdown("## 📋 All Parsed Data")
    
    successful = [r for r in results if r['status'] == 'success']
    
    if successful:
        st.info(f"Showing {len(successful)} successful analyses. Download the Excel report for complete data including all Q-A pairs.")
        
        # Tampilkan data lengkap
        raw_data = []
        for result in successful:
            special_notes = []
            if result.get('customer_leave'):
                special_notes.append("Customer Leave")
            if result.get('follow_up_ticket'):
                special_notes.append("Follow-up")
            
            raw_data.append({
                'Ticket ID': result['ticket_id'],
                'Main Question': result['main_question'],
                'Main Question Time': result.get('main_question_time'),
                'Issue Type': result['final_issue_type'],
                'First Reply Found': result['first_reply_found'],
                'First Reply Message': result.get('first_reply_message', '')[:100] + '...' if result.get('first_reply_message') else 'Not found',
                'First Reply Time': result.get('first_reply_time'),
                'First Reply LT (min)': result.get('first_reply_lead_time_minutes'),
                'Final Reply Found': result['final_reply_found'],
                'Final Reply Message': result.get('final_reply_message', '')[:100] + '...' if result.get('final_reply_message') else 'Not found',
                'Final Reply Time': result.get('final_reply_time'),
                'Final Reply LT (min)': result.get('final_reply_lead_time_minutes'),
                'Performance': result['performance_rating'],
                'Quality Score': result['quality_score'],
                'Special Notes': ', '.join(special_notes) if special_notes else 'None',
                'Total Messages': result['total_messages'],
                'Total QA Pairs': result['total_qa_pairs']
            })
        
        df_raw = pd.DataFrame(raw_data)
        st.dataframe(df_raw, use_container_width=True)
        
        # Data summary
        st.markdown("### 📊 Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tickets", len(successful))
        with col2:
            avg_quality = np.mean([r['quality_score'] for r in successful])
            st.metric("Avg Quality Score", f"{avg_quality:.1f}")
        with col3:
            first_reply_rate = sum(1 for r in successful if r['first_reply_found']) / len(successful) * 100
            st.metric("First Reply Rate", f"{first_reply_rate:.1f}%")
        with col4:
            final_reply_rate = sum(1 for r in successful if r['final_reply_found']) / len(successful) * 100
            st.metric("Final Reply Rate", f"{final_reply_rate:.1f}%")
        
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
    
    # System information
    st.markdown("### 🔧 System Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Python Version", sys.version.split()[0])
    with col2:
        st.metric("Pandas Version", pd.__version__)
    with col3:
        st.metric("Analysis Time", f"{stats.get('analysis_duration_seconds', 0):.1f}s")

# Main execution
if __name__ == "__main__":
    if not ANALYSIS_AVAILABLE:
        st.error("""
        ❌ Analysis modules not available!
        
        Please ensure:
        1. File `Chat_Analyzer_System.py` exists in the same directory
        2. File tersebut berisi semua class: DataPreprocessor, CompleteAnalysisPipeline, etc.
        3. Dependencies sudah terinstall
        """)
        st.stop()
    
    # Check if analysis is complete
    if st.session_state.get('analysis_complete', False):
        display_complete_results()
    else:
        main_interface()







