# app.py - FIXED Streamlit Dashboard dengan Logic Baru
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

# Import analysis modules
sys.path.append('.')
try:
    from Chat_Analyzer_System import (
        DataPreprocessor, CompleteAnalysisPipeline, 
        ResultsExporter, ModelTrainer, Config
    )
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Analysis modules not available: {e}")
    ANALYSIS_AVAILABLE = False

st.set_page_config(
    page_title="Toyota Chat Analysis Dashboard",
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
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 5px;
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
    .download-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 20px 0;
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
    st.markdown('<h1 class="main-header">🤖 Toyota Chat Lead Time Analysis</h1>', unsafe_allow_html=True)
    st.markdown("Upload your chat data Excel file untuk analisis performa customer service")
    
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
        1. Upload Excel file dengan data chat
        2. Atur pengaturan analisis  
        3. Klik 'Start Analysis'
        4. Lihat hasil & download reports
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
        # Show sample data option
        st.markdown("---")
        st.markdown("## 🎯 Try with Sample Data")
        
        if os.path.exists("data/raw_conversation.xlsx"):
            if st.button("🧪 Analyze Sample Data", use_container_width=True):
                with st.spinner("Loading sample data..."):
                    try:
                        # Simulate file upload for sample data
                        class MockFile:
                            def __init__(self):
                                self.name = "sample_data.xlsx"
                        uploaded_file = MockFile()
                        results, stats, excel_path = run_analysis(uploaded_file, 50, "Quick Analysis")
                        
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
        else:
            st.info("📁 No sample data found. Please upload your own Excel file.")

def run_analysis(uploaded_file, max_tickets, analysis_type):
    """Run analysis pada uploaded file"""
    
    # Save uploaded file ke temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
        # Handle both real uploaded files and sample data
        if hasattr(uploaded_file, 'getvalue'):
            tmp_file.write(uploaded_file.getvalue())
        else:
            # For sample data, copy from actual file
            import shutil
            if os.path.exists("data/raw_conversation.xlsx"):
                shutil.copy2("data/raw_conversation.xlsx", tmp_file.name)
            else:
                st.error("Sample data file not found")
                return None, None, None
        temp_path = tmp_file.name
    
    try:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Loading data...")
        progress_bar.progress(10)
        
        # Load dan preprocess data
        preprocessor = DataPreprocessor()
        df = preprocessor.load_raw_data(temp_path)
        
        if df is None:
            st.error("❌ Failed to load data. Please check file format.")
            return None, None, None
        
        status_text.text("🔍 Analyzing conversations...")
        progress_bar.progress(30)
        
        # Run analysis dengan error handling
        try:
            pipeline = CompleteAnalysisPipeline()
            results, stats = pipeline.analyze_all_tickets(df, max_tickets=max_tickets)
        except Exception as e:
            st.error(f"❌ Analysis error: {str(e)}")
            st.error("Please check your data format and try again.")
            return None, None, None
        
        status_text.text("💾 Exporting results...")
        progress_bar.progress(70)
        
        # Export results dengan error handling
        try:
            exporter = ResultsExporter()
            excel_path = exporter.export_comprehensive_results(results, stats)
        except Exception as e:
            st.warning(f"⚠️ Excel export had issues: {e}")
            excel_path = None
        
        # Fallback jika export gagal
        if excel_path is None or not os.path.exists(excel_path):
            st.warning("⚠️ Excel export had issues, creating basic export...")
            try:
                # Create basic export
                export_data = []
                for result in results:
                    if result['status'] == 'success':
                        export_data.append({
                            'ticket_id': result['ticket_id'],
                            'issue_type': result['final_issue_type'],
                            'main_question': result['main_question'],
                            'first_reply_found': result['first_reply_found'],
                            'final_reply_found': result['final_reply_found'],
                            'first_reply_lead_time_min': result.get('first_reply_lead_time_minutes'),
                            'final_reply_lead_time_min': result.get('final_reply_lead_time_minutes'),
                            'performance_rating': result['performance_rating'],
                            'customer_leave': result.get('customer_leave', False),
                            'follow_up_ticket': result.get('follow_up_ticket', '')
                        })
                
                df_export = pd.DataFrame(export_data)
                excel_path = f"output/analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                df_export.to_excel(excel_path, index=False)
                st.success(f"✅ Basic export created: {excel_path}")
            except Exception as e:
                st.error(f"❌ Basic export also failed: {e}")
                excel_path = None
        
        status_text.text("📊 Creating visualizations...")
        progress_bar.progress(90)
        
        # Run model training untuk comprehensive analysis
        if analysis_type == "Comprehensive Analysis":
            try:
                model_trainer = ModelTrainer(pipeline)
                training_data = model_trainer.collect_training_data_from_analysis(results)
                model_trainer.enhance_training_data()
                model_trainer.train_and_evaluate_model(test_size=0.2)
            except Exception as e:
                st.warning(f"Model training skipped: {e}")
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return results, stats, excel_path
        
    except Exception as e:
        st.error(f"❌ Analysis error: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None, None, None
    finally:
        # Cleanup temporary file
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
            
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
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Tickets", stats.get('total_tickets', 0))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        success_rate = stats.get('success_rate', 0) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'overall_lead_times' in stats:
            avg_lead_time = stats['overall_lead_times'].get('final_reply_avg_minutes', 0)
            st.metric("Avg Final Reply", f"{avg_lead_time:.1f} min")
        else:
            st.metric("Avg Lead Time", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'performance_distribution' in stats:
            excellent = stats['performance_distribution'].get('excellent', 0)
            st.metric("Excellent", excellent)
        else:
            st.metric("Excellent", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'issue_type_distribution' in stats:
            total_issues = sum(stats['issue_type_distribution'].values())
            st.metric("Issues Found", total_issues)
        else:
            st.metric("Issues Found", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="download-section">', unsafe_allow_html=True)
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
    """Display detailed information for a single ticket - DIPERBAIKI"""
    # Main Question
    st.markdown("### 📝 Main Question")
    st.markdown(f'<div class="message-box"><strong>Question:</strong> {result["main_question"]}</div>', unsafe_allow_html=True)
    st.caption(f"Detected as: {result['final_issue_type'].upper()} (Confidence: {result['detection_confidence']:.2f})")
    
    # Special Cases Notes - LOGIC BARU
    special_notes = []
    if result.get('customer_leave'):
        special_notes.append("🚶 **Customer Leave**: Customer left conversation without response")
    if result.get('follow_up_ticket'):
        special_notes.append(f"🔄 **Follow-up**: Resolved in ticket {result['follow_up_ticket']}")
    if result.get('customer_leave_note'):
        special_notes.append(f"📝 **Note**: {result['customer_leave_note']}")
    
    if special_notes:
        st.markdown("### 🚨 Special Conditions")
        for note in special_notes:
            st.markdown(f'<div class="special-case">{note}</div>', unsafe_allow_html=True)
    
    # First Reply Section - LOGIC BARU
    st.markdown("#### 🔄 First Reply Analysis")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if result['first_reply_found']:
            first_reply_msg = result.get('first_reply_message', 'No message content available')
            st.markdown(f'<div class="message-box"><strong>First Reply:</strong> {first_reply_msg}</div>', unsafe_allow_html=True)
            
            # Tampilkan note jika ada
            if result.get('first_reply_note'):
                st.info(f"Note: {result['first_reply_note']}")
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
    
    # Final Reply Section - LOGIC BARU  
    st.markdown("#### ✅ Final Reply Analysis")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if result['final_reply_found']:
            final_reply_msg = result.get('final_reply_message', 'No message content available')
            st.markdown(f'<div class="message-box"><strong>Final Reply:</strong> {final_reply_msg}</div>', unsafe_allow_html=True)
            
            # Tampilkan note khusus
            if result.get('customer_leave_note'):
                st.warning(result['customer_leave_note'])
            elif result.get('final_reply_note'):
                st.info(f"Note: {result['final_reply_note']}")
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
    
    # Requirement Compliance - LOGIC BARU
    st.markdown("#### 📋 Requirement Compliance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if result['requirement_compliant']:
            st.success("✅ Requirements Met")
        else:
            st.warning("⚠️ Requirements Partially Met")
    
    with col2:
        st.metric("First Reply", "✅" if result['first_reply_found'] else "❌")
    
    with col3:
        st.metric("Final Reply", "✅" if result['final_reply_found'] else "❌")
    
    # Recommendation
    if result.get('recommendation'):
        st.info(f"💡 **Recommendation**: {result['recommendation']}")

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
            
            # Prepare data for visualization - also maintain the same order
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
                
                # Ensure the x-axis order in the chart
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
                    category_orders={"Issue Type": [t.upper() for t in desired_order]}  # This ensures the order in the chart
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
            
            if len(perf_pivot) > 1:  # Only plot if we have multiple issue types
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
    """Display special cases analysis (customer leave & follow-up)"""
    st.markdown("## 🚨 Special Cases Analysis")
    
    successful = [r for r in results if r['status'] == 'success']
    
    # Customer Leave Cases
    customer_leave_cases = [r for r in successful if r.get('customer_leave')]
    follow_up_cases = [r for r in successful if r.get('follow_up_ticket')]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚶 Customer Leave Cases")
        if customer_leave_cases:
            st.info(f"Found {len(customer_leave_cases)} conversations where customer left without response")
            
            leave_data = []
            for result in customer_leave_cases:
                leave_data.append({
                    'Ticket ID': result['ticket_id'],
                    'Issue Type': result['final_issue_type'],
                    'Main Question': result['main_question'][:80] + '...',
                    'Final Reply Used': '✅' if result['final_reply_found'] else '❌',
                    'Performance': result['performance_rating'].upper()
                })
            
            df_leave = pd.DataFrame(leave_data)
            st.dataframe(df_leave, use_container_width=True)
        else:
            st.success("✅ No customer leave cases detected")
    
    with col2:
        st.markdown("### 🔄 Follow-up Cases")
        if follow_up_cases:
            st.info(f"Found {len(follow_up_cases)} issues resolved in different tickets")
            
            follow_up_data = []
            for result in follow_up_cases:
                follow_up_data.append({
                    'Original Ticket': result['ticket_id'],
                    'Follow-up Ticket': result['follow_up_ticket'],
                    'Issue Type': result['final_issue_type'],
                    'Main Question': result['main_question'][:80] + '...',
                    'Performance': result['performance_rating'].upper()
                })
            
            df_follow_up = pd.DataFrame(follow_up_data)
            st.dataframe(df_follow_up, use_container_width=True)
        else:
            st.success("✅ No follow-up cases detected")
    
    # Analysis of special cases impact
    if customer_leave_cases or follow_up_cases:
        st.markdown("### 📊 Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if customer_leave_cases:
                # Performance distribution for customer leave cases
                perf_counts = pd.Series([r['performance_rating'] for r in customer_leave_cases]).value_counts()
                fig_leave_perf = px.pie(
                    values=perf_counts.values,
                    names=perf_counts.index,
                    title='Performance Distribution - Customer Leave Cases',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_leave_perf, use_container_width=True)
        
        with col2:
            if follow_up_cases:
                # Issue type distribution for follow-up cases
                issue_counts = pd.Series([r['final_issue_type'] for r in follow_up_cases]).value_counts()
                fig_follow_up_issues = px.pie(
                    values=issue_counts.values,
                    names=issue_counts.index,
                    title='Issue Type Distribution - Follow-up Cases',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_follow_up_issues, use_container_width=True)

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
        for result in failed[:5]:  # Show first 5 failures
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
