# main.py - Enhanced Main Execution & Deployment
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Import semua modul yang sudah ada
sys.path.append('.')  # Pastikan bisa import dari current directory

def main():
    """Main execution function - Complete end-to-end analysis"""
    print("🚀 CHAT ANALYSIS SYSTEM - MAIN EXECUTION")
    print("=" * 60)
    
    try:
        # Step 1: Load data
        print("📁 Step 1: Loading data...")
        from Chat_Analyzer_System import DataPreprocessor, Config
        
        preprocessor = DataPreprocessor()
        df = preprocessor.load_raw_data(Config.RAW_DATA_PATH)
        
        if df is None:
            print("❌ Failed to load data")
            return
        
        print(f"✅ Data loaded: {len(df)} rows, {df['Ticket Number'].nunique()} tickets")
        
        # Step 2: Run complete analysis
        print("🔍 Step 2: Running complete analysis...")
        from Chat_Analyzer_System import CompleteAnalysisPipeline
        
        pipeline = CompleteAnalysisPipeline()
        results, stats = pipeline.analyze_all_tickets(df, max_tickets=100000)  
        
        # Step 3: Export results
        print("💾 Step 3: Exporting results...")
        from Chat_Analyzer_System import ResultsExporter
        
        exporter = ResultsExporter()
        excel_file = exporter.export_comprehensive_results(results, stats)
        
        # Step 4: Create visualizations
        print("📊 Step 4: Creating visualizations...")
        exporter.create_comprehensive_visualizations(results, stats)
        
        # Step 5: Generate model evaluation report
        print("🤖 Step 5: Generating model evaluation...")
        from Chat_Analyzer_System import ModelTrainer
        
        model_trainer = ModelTrainer(pipeline)
        training_data = model_trainer.collect_training_data_from_analysis(results)
        model_trainer.enhance_training_data()
        accuracy = model_trainer.train_and_evaluate_model(test_size=0.2)
        model_trainer.save_model_report()
        
        # Step 6: Generate PDF Summary Report
        print("📄 Step 6: Generating PDF summary...")
        try:
            from Chat_Analyzer_System import PDFReportGenerator
            pdf_generator = PDFReportGenerator()
            pdf_path = pdf_generator.generate_management_summary(results, stats)
            print(f"✅ PDF report generated: {pdf_path}")
        except Exception as e:
            print(f"⚠️ PDF generation failed: {e}")
        
        # Step 7: Display final summary
        print("\n🎉 MAIN EXECUTION COMPLETED!")
        print("=" * 60)
        print(f"📈 FINAL RESULTS SUMMARY:")
        print(f"   • Tickets Processed: {stats['total_tickets']}")
        print(f"   • Successful Analysis: {stats['successful_analysis']} ({stats['success_rate']*100:.1f}%)")
        
        if 'lead_time_by_issue_type' in stats:
            print(f"\n⏱️ AVERAGE LEAD TIMES:")
            for issue_type, lt_stats in stats['lead_time_by_issue_type'].items():
                print(f"   • {issue_type.title()}: {lt_stats['avg_lead_time']:.2f} min")
        
        if 'performance_distribution' in stats:
            excellent = stats['performance_distribution'].get('excellent', 0)
            print(f"   • Excellent Performance: {excellent} tickets")
        
        if 'conversation_type_distribution' in stats:
            print(f"\n🏷️ CONVERSATION TYPES:")
            for conv_type, count in stats['conversation_type_distribution'].items():
                percentage = (count / stats['successful_analysis']) * 100
                print(f"   • {conv_type.title()}: {count} ({percentage:.1f}%)")
        
        if 'issue_type_distribution' in stats:
            print(f"\n🎯 ISSUE TYPES:")
            for issue_type, count in stats['issue_type_distribution'].items():
                percentage = (count / stats['successful_analysis']) * 100
                print(f"   • {issue_type.title()}: {count} ({percentage:.1f}%)")
        
        # Deployment instructions
        print("\n🚀 DEPLOYMENT INSTRUCTIONS:")
        print("1. Streamlit Dashboard: streamlit run app.py")
        print("2. Model Files: Check 'models/' folder")
        print("3. Results: Check 'output/' folder")
        print("4. Analysis Report: output/comprehensive_analysis_results.xlsx")
        print("5. PDF Summary: output/reports/management_summary.pdf")
        
        return results, stats
        
    except Exception as e:
        print(f"❌ Main execution failed: {e}")
        return None, None

def deploy_streamlit():
    """Deploy Streamlit dashboard"""
    print("\n🎨 DEPLOYING STREAMLIT DASHBOARD...")
    
    streamlit_code = '''
# app.py - FIXED Streamlit Dashboard dengan Lead Time per Issue Type
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
            help="Quick: Basic metrics, Comprehensive: Full analysis with ML"
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
        
        # Run analysis
        pipeline = CompleteAnalysisPipeline()
        results, stats = pipeline.analyze_all_tickets(df, max_tickets=max_tickets)
        
        status_text.text("💾 Exporting results...")
        progress_bar.progress(70)
        
        # Export results
        exporter = ResultsExporter()
        excel_path = exporter.export_comprehensive_results(results, stats)
        
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
                            'performance_rating': result['performance_rating']
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
            os.unlink(temp_path)

def display_complete_results():
    """Display COMPLETE analysis results dengan semua tab dan download"""
    
    results = st.session_state.analysis_results
    stats = st.session_state.analysis_stats
    
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
        if 'lead_time_stats' in stats:
            avg_lead_time = stats['lead_time_stats']['avg_lead_time_minutes']
            st.metric("Avg Lead Time", f"{avg_lead_time:.1f} min")
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

    # DOWNLOAD SECTION - PALING PENTING!
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
        st.success("✅ Excel report contains ALL parsed data: Q-A pairs, main issues, reply analysis, timestamps, and detailed metrics!")
    else:
        st.error("❌ Excel file not available for download. Analysis may not have exported properly.")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # TABS - SEMUA FITUR ANALYSIS
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview", "🎯 Main Issues", "⏱️ Lead Times", "📊 Performance", "💬 Conversation Types", "📋 All Data"
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
        display_conversation_types_tab(results, stats)
    
    with tab6:
        display_raw_data_tab(results)

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
    if 'lead_time_stats' in stats:
        lt_stats = stats['lead_time_stats']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Lead Time", f"{lt_stats['avg_lead_time_minutes']:.1f} min")
        with col2:
            st.metric("Median Lead Time", f"{lt_stats['median_lead_time_minutes']:.1f} min")
        with col3:
            st.metric("Min Lead Time", f"{lt_stats['min_lead_time_minutes']:.1f} min")
        with col4:
            st.metric("Max Lead Time", f"{lt_stats['max_lead_time_minutes']:.1f} min")

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
            display_data.append({
                'Ticket ID': result['ticket_id'],
                'Main Question': result['main_question'][:80] + '...' if len(result['main_question']) > 80 else result['main_question'],
                'Issue Type': result['final_issue_type'].upper(),
                'First Reply': '✅' if result['first_reply_found'] else '❌',
                'Final Reply': '✅' if result['final_reply_found'] else '❌',
                'First Reply LT (min)': result.get('first_reply_lead_time_minutes', 'N/A'),
                'Final Reply LT (min)': result.get('final_reply_lead_time_minutes', 'N/A'),
                'Performance': result['performance_rating'].upper(),
                'Quality Score': result['quality_score']
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
    
    # First Reply Section
    st.markdown("#### 🔄 First Reply Analysis")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if result['first_reply_found']:
            first_reply_msg = result.get('first_reply_message', 'No message content available')
            st.markdown(f'<div class="message-box"><strong>First Reply:</strong> {first_reply_msg}</div>', unsafe_allow_html=True)
        else:
            st.error("❌ No first reply found")
            st.markdown('<div class="message-box">First reply was not detected for this conversation.</div>', unsafe_allow_html=True)
    
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
            st.markdown(f'<div class="message-box"><strong>Final Reply:</strong> {final_reply_msg}</div>', unsafe_allow_html=True)
        else:
            st.error("❌ No final reply found")
            st.markdown('<div class="message-box">Final reply was not detected for this conversation.</div>', unsafe_allow_html=True)
    
    with col2:
        if result['final_reply_found']:
            st.metric("Lead Time", f"{result.get('final_reply_lead_time_minutes', 'N/A')} min")
            st.metric("Time Format", result.get('final_reply_lead_time_hhmmss', 'N/A'))
        else:
            st.metric("Status", "Not Found")
    
    # All Q-A Pairs (Raw parsing results)
    if '_raw_qa_pairs' in result:
        st.markdown("### 📋 All Question-Answer Pairs")
        qa_pairs = result['_raw_qa_pairs']
        st.info(f"Found {len(qa_pairs)} Q-A pairs in this conversation")
        
        for i, qa_pair in enumerate(qa_pairs):
            with st.expander(f"Q-A Pair {i+1} - {'✅ Answered' if qa_pair['is_answered'] else '❌ Unanswered'}"):
                col_q, col_a = st.columns(2)
                
                with col_q:
                    st.markdown("**Question:**")
                    st.markdown(f'<div class="message-box">{qa_pair["question"]}</div>', unsafe_allow_html=True)
                    st.caption(f"Time: {qa_pair.get('question_time', 'N/A')}")
                    st.caption(f"Bubbles: {qa_pair.get('bubble_count', 1)}")
                
                with col_a:
                    if qa_pair['is_answered']:
                        st.markdown("**Answer:**")
                        st.markdown(f'<div class="message-box">{qa_pair["answer"]}</div>', unsafe_allow_html=True)
                        st.caption(f"Time: {qa_pair.get('answer_time', 'N/A')}")
                        st.caption(f"Role: {qa_pair.get('answer_role', 'N/A')}")
                        st.caption(f"Lead Time: {qa_pair.get('lead_time_minutes', 'N/A')} min")
                    else:
                        st.warning("No answer found")
    
    # Performance Metrics
    st.markdown("### 📊 Performance Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Performance", result['performance_rating'].upper())
    with col2:
        st.metric("Quality Score", f"{result['quality_score']}/6")
    with col3:
        st.metric("Issue Type", result['final_issue_type'].upper())
    with col4:
        st.metric("Total Messages", result['total_messages'])

def display_lead_time_tab(results, stats):
    """Display lead time analysis dengan breakdown per issue type"""
    st.markdown("## ⏱️ Lead Time Analysis")
    
    successful = [r for r in results if r['status'] == 'success']
    
    # Calculate lead times per issue type
    issue_type_lead_times = {}
    
    for result in successful:
        issue_type = result['final_issue_type']
        first_lt = result.get('first_reply_lead_time_minutes')
        final_lt = result.get('final_reply_lead_time_minutes')
        
        if issue_type not in issue_type_lead_times:
            issue_type_lead_times[issue_type] = {
                'first_reply_times': [],
                'final_reply_times': []
            }
        
        if first_lt is not None:
            issue_type_lead_times[issue_type]['first_reply_times'].append(first_lt)
        if final_lt is not None:
            issue_type_lead_times[issue_type]['final_reply_times'].append(final_lt)
    
    # Display average lead times per issue type
    st.markdown("### 📊 Average Lead Times by Issue Type")
    
    if issue_type_lead_times:
        # Create summary table
        summary_data = []
        for issue_type, times in issue_type_lead_times.items():
            first_avg = np.mean(times['first_reply_times']) if times['first_reply_times'] else None
            final_avg = np.mean(times['final_reply_times']) if times['final_reply_times'] else None
            first_count = len(times['first_reply_times'])
            final_count = len(times['final_reply_times'])
            
            summary_data.append({
                'Issue Type': issue_type.upper(),
                'First Reply Avg (min)': f"{first_avg:.1f}" if first_avg is not None else "N/A",
                'Final Reply Avg (min)': f"{final_avg:.1f}" if final_avg is not None else "N/A",
                'First Reply Samples': first_count,
                'Final Reply Samples': final_count
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
        
        # Visual comparison
        st.markdown("### 📈 Lead Time Comparison by Issue Type")
        
        viz_data = []
        for issue_type, times in issue_type_lead_times.items():
            first_avg = np.mean(times['first_reply_times']) if times['first_reply_times'] else None
            final_avg = np.mean(times['final_reply_times']) if times['final_reply_times'] else None
            
            if first_avg is not None:
                viz_data.append({
                    'Issue Type': issue_type.upper(),
                    'Lead Time Type': 'First Reply',
                    'Average Lead Time (min)': first_avg,
                    'Samples': len(times['first_reply_times'])
                })
            if final_avg is not None:
                viz_data.append({
                    'Issue Type': issue_type.upper(),
                    'Lead Time Type': 'Final Reply',
                    'Average Lead Time (min)': final_avg,
                    'Samples': len(times['final_reply_times'])
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
                }
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
        
        # Distribution charts per issue type
        st.markdown("### 📊 Lead Time Distribution by Issue Type")
        
        for issue_type, times in issue_type_lead_times.items():
            if times['first_reply_times'] or times['final_reply_times']:
                st.markdown(f"#### {issue_type.upper()} Issues")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if times['first_reply_times']:
                        fig_first = px.histogram(
                            x=times['first_reply_times'],
                            title=f'{issue_type.upper()} - First Reply Lead Time Distribution',
                            labels={'x': 'Lead Time (minutes)', 'y': 'Frequency'},
                            nbins=15,
                            color_discrete_sequence=['#2E86AB']
                        )
                        fig_first.add_vline(
                            x=np.mean(times['first_reply_times']), 
                            line_dash="dash", 
                            line_color="red",
                            annotation_text=f"Mean: {np.mean(times['first_reply_times']):.1f} min"
                        )
                        st.plotly_chart(fig_first, use_container_width=True)
                    else:
                        st.info(f"No first reply data for {issue_type} issues")
                
                with col2:
                    if times['final_reply_times']:
                        fig_final = px.histogram(
                            x=times['final_reply_times'],
                            title=f'{issue_type.upper()} - Final Reply Lead Time Distribution',
                            labels={'x': 'Lead Time (minutes)', 'y': 'Frequency'},
                            nbins=15,
                            color_discrete_sequence=['#A23B72']
                        )
                        fig_final.add_vline(
                            x=np.mean(times['final_reply_times']), 
                            line_dash="dash", 
                            line_color="red",
                            annotation_text=f"Mean: {np.mean(times['final_reply_times']):.1f} min"
                        )
                        st.plotly_chart(fig_final, use_container_width=True)
                    else:
                        st.info(f"No final reply data for {issue_type} issues")
    else:
        st.info("No lead time data available for analysis")

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
            perf_metrics.append({
                'Ticket ID': result['ticket_id'],
                'Issue Type': result['final_issue_type'],
                'Performance': result['performance_rating'].upper(),
                'Quality Score': result['quality_score'],
                'First Reply LT': result.get('first_reply_lead_time_minutes', 'N/A'),
                'Final Reply LT': result.get('final_reply_lead_time_minutes', 'N/A'),
                'First Reply': '✅' if result['first_reply_found'] else '❌',
                'Final Reply': '✅' if result['final_reply_found'] else '❌'
            })
        
        df_perf_metrics = pd.DataFrame(perf_metrics)
        st.dataframe(df_perf_metrics, use_container_width=True)

def display_conversation_types_tab(results, stats):
    """Display conversation type analysis"""
    st.markdown("## 💬 Conversation Type Analysis")
    
    successful = [r for r in results if r['status'] == 'success']
    
    if successful:
        # Calculate conversation types based on reply patterns
        conversation_types = []
        for result in successful:
            if not result['first_reply_found'] and not result['final_reply_found']:
                conv_type = 'no_reply'
            elif result['first_reply_found'] and not result['final_reply_found']:
                conv_type = 'abandoned'
            elif not result['first_reply_found'] and result['final_reply_found']:
                conv_type = 'direct_final'
            else:
                conv_type = 'complete'
            
            conversation_types.append(conv_type)
        
        if conversation_types:
            conv_counts = pd.Series(conversation_types).value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Conversation Type Distribution
                fig_conv = px.pie(
                    values=conv_counts.values, 
                    names=conv_counts.index,
                    title='Conversation Type Distribution',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_conv, use_container_width=True)
            
            with col2:
                # Conversation Type Bar Chart
                fig_conv_bar = px.bar(
                    x=conv_counts.index, 
                    y=conv_counts.values,
                    title='Conversation Types',
                    labels={'x': 'Conversation Type', 'y': 'Count'},
                    color=conv_counts.index,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig_conv_bar, use_container_width=True)
            
            # Conversation type details
            st.markdown("### 🔍 Conversation Type Details")
            conv_data = []
            for result, conv_type in zip(successful, conversation_types):
                conv_data.append({
                    'Ticket ID': result['ticket_id'],
                    'Conversation Type': conv_type.replace('_', ' ').title(),
                    'First Reply': '✅' if result['first_reply_found'] else '❌',
                    'Final Reply': '✅' if result['final_reply_found'] else '❌',
                    'Performance': result['performance_rating'].upper(),
                    'Main Question': result['main_question'][:60] + '...' if len(result['main_question']) > 60 else result['main_question']
                })
            
            df_conv = pd.DataFrame(conv_data)
            st.dataframe(df_conv, use_container_width=True)

def display_raw_data_tab(results):
    """Display raw data tab untuk melihat semua hasil parse"""
    st.markdown("## 📋 All Parsed Data")
    
    successful = [r for r in results if r['status'] == 'success']
    
    if successful:
        st.info(f"Showing {len(successful)} successful analyses. Download the Excel report for complete data including all Q-A pairs.")
        
        # Tampilkan data lengkap
        raw_data = []
        for result in successful:
            raw_data.append({
                'Ticket ID': result['ticket_id'],
                'Main Question': result['main_question'],
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

# ===== DEBUG FUNCTIONS UNTUK STREAMLIT =====
def debug_timestamp_issues(results):
    """Debug function untuk investigasi timestamp issues"""
    st.markdown("---")
    st.markdown("## 🐛 Debug Timestamp Issues")
    
    problematic_tickets = []
    
    for result in results:
        if result['status'] == 'success':
            first_lt = result.get('first_reply_lead_time_minutes')
            final_lt = result.get('final_reply_lead_time_minutes')
            
            # Cari tickets dengan lead time negatif
            if (first_lt is not None and first_lt < 0) or (final_lt is not None and final_lt < 0):
                problematic_tickets.append({
                    'ticket_id': result['ticket_id'],
                    'main_question': result['main_question'][:50] + '...',
                    'main_question_time': result.get('main_question_time'),
                    'first_reply_time': result.get('first_reply_time'),
                    'final_reply_time': result.get('final_reply_time'),
                    'first_lead_time': first_lt,
                    'final_lead_time': final_lt
                })
    
    if problematic_tickets:
        st.error(f"❌ Found {len(problematic_tickets)} tickets with negative lead times")
        df_problems = pd.DataFrame(problematic_tickets)
        st.dataframe(df_problems, use_container_width=True)
        
        # Show raw data untuk ticket pertama yang problematic
        if problematic_tickets:
            st.markdown("### 🔍 Raw Data for Problematic Ticket")
            ticket_id = problematic_tickets[0]['ticket_id']
            problematic_result = next((r for r in results if r['ticket_id'] == ticket_id), None)
            
            if problematic_result and '_raw_qa_pairs' in problematic_result:
                st.write("**All Q-A Pairs:**")
                for i, qa_pair in enumerate(problematic_result['_raw_qa_pairs']):
                    st.write(f"**Pair {i+1}:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"Q: {qa_pair.get('question', 'N/A')}")
                        st.write(f"Q Time: {qa_pair.get('question_time', 'N/A')}")
                    with col2:
                        if qa_pair.get('is_answered'):
                            st.write(f"A: {qa_pair.get('answer', 'N/A')}")
                            st.write(f"A Time: {qa_pair.get('answer_time', 'N/A')}")
                            st.write(f"Lead Time: {qa_pair.get('lead_time_minutes', 'N/A')} min")
    else:
        st.success("✅ No timestamp issues found")

def debug_message_sequence(results):
    """Debug function untuk melihat urutan message sebenarnya"""
    st.markdown("## 🔍 Debug Message Sequence")
    
    successful = [r for r in results if r['status'] == 'success']
    
    if successful:
        ticket_options = [f"{r['ticket_id']} - {r['main_question'][:50]}..." for r in successful]
        selected_ticket = st.selectbox("Select ticket to debug sequence:", ticket_options, key="debug_sequence")
        
        if selected_ticket:
            ticket_id = selected_ticket.split(' - ')[0]
            selected_result = next((r for r in successful if r['ticket_id'] == ticket_id), None)
            
            if selected_result and '_raw_qa_pairs' in selected_result:
                # Tampilkan Q-A pairs dengan urutan asli
                st.markdown("### 📋 Q-A Pairs (Current Order)")
                qa_pairs = selected_result['_raw_qa_pairs']
                
                for i, qa_pair in enumerate(qa_pairs):
                    with st.expander(f"Q-A Pair {i+1} - Position {qa_pair.get('position', 'N/A')} - {'✅ Answered' if qa_pair['is_answered'] else '❌ Unanswered'}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Question:**")
                            st.markdown(f'<div class="message-box">{qa_pair["question"]}</div>', unsafe_allow_html=True)
                            question_time = qa_pair.get('question_time')
                            if question_time:
                                st.caption(f"🕒 **Time:** {question_time}")
                        
                        with col2:
                            if qa_pair['is_answered']:
                                st.markdown("**Answer:**")
                                st.markdown(f'<div class="message-box">{qa_pair["answer"]}</div>', unsafe_allow_html=True)
                                answer_time = qa_pair.get('answer_time')
                                if answer_time:
                                    st.caption(f"🕒 **Time:** {answer_time}")
                                st.caption(f"⏱️ **Lead Time:** {qa_pair.get('lead_time_minutes', 'N/A')} min")
                            else:
                                st.warning("No answer found")
                
                # Tampilkan urutan yang seharusnya (sorted by time)
                st.markdown("### 🔄 Correct Chronological Order")
                sorted_qa_pairs = sorted(qa_pairs, key=lambda x: x.get('question_time') if x.get('question_time') else pd.Timestamp.min)
                
                for i, qa_pair in enumerate(sorted_qa_pairs):
                    with st.expander(f"Chronological {i+1} - Original Position {qa_pair.get('position', 'N/A')} - {'✅ Answered' if qa_pair['is_answered'] else '❌ Unanswered'}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Question:**")
                            st.markdown(f'<div class="message-box">{qa_pair["question"]}</div>', unsafe_allow_html=True)
                            question_time = qa_pair.get('question_time')
                            if question_time:
                                st.caption(f"🕒 **Time:** {question_time}")
                        
                        with col2:
                            if qa_pair['is_answered']:
                                st.markdown("**Answer:**")
                                st.markdown(f'<div class="message-box">{qa_pair["answer"]}</div>', unsafe_allow_html=True)
                                answer_time = qa_pair.get('answer_time')
                                if answer_time:
                                    st.caption(f"🕒 **Time:** {answer_time}")
                                st.caption(f"⏱️ **Lead Time:** {qa_pair.get('lead_time_minutes', 'N/A')} min")
                            else:
                                st.warning("No answer found")
                
                # Summary
                st.markdown("### 📊 Sequence Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Q-A Pairs", len(qa_pairs))
                    st.metric("Answered Pairs", sum(1 for p in qa_pairs if p['is_answered']))
                
                with col2:
                    # Check if sequence is correct
                    positions = [p.get('position', 0) for p in qa_pairs]
                    is_sorted = positions == sorted(positions)
                    st.metric("Sequence Correct", "✅" if is_sorted else "❌")
                    
                    times = [p.get('question_time') for p in qa_pairs if p.get('question_time')]
                    time_sorted = times == sorted(times) if len(times) > 1 else True
                    st.metric("Time Sorted", "✅" if time_sorted else "❌")
'''
    
    # Save Streamlit app
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(streamlit_code)
    
    print("✅ Streamlit dashboard created: app.py")
    print("💡 Run: streamlit run app.py")

def create_deployment_package():
    """Create deployment package dengan semua files yang diperlukan"""
    print("\n📦 CREATING DEPLOYMENT PACKAGE...")
    
    # Create requirements.txt
    requirements = """
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
openpyxl>=3.0.0
joblib>=1.0.0
streamlit>=1.0.0
python-dateutil>=2.8.0
reportlab>=3.6.0
streamlit-aggrid>=0.3.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    # Create PDF report generator
    pdf_code = '''
# pdf_report_generator.py - PDF Report Generator
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
import os

class PDFReportGenerator:
    def __init__(self):
        self.output_dir = "output/reports/"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_management_summary(self, results, stats):
        """Generate PDF summary report untuk management"""
        filename = f"{self.output_dir}management_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        doc = SimpleDocTemplate(filename, pagesize=A4)
        story = []
        
        styles = getSampleStyleSheet()
        title_style = styles['Heading1']
        heading_style = styles['Heading2']
        normal_style = styles['BodyText']
        
        # Title
        story.append(Paragraph("Chat Analysis - Management Summary", title_style))
        story.append(Spacer(1, 12))
        
        # Timestamp
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Tickets Processed', str(stats.get('total_tickets', 0))],
            ['Successful Analysis', f"{stats.get('successful_analysis', 0)} ({stats.get('success_rate', 0)*100:.1f}%)"],
            ['Analysis Duration', f"{stats.get('analysis_duration_seconds', 0):.1f} seconds"]
        ]
        
        if 'issue_type_distribution' in stats:
            for issue_type, count in stats['issue_type_distribution'].items():
                summary_data.append([f'{issue_type.title()} Issues', str(count)])
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Performance Metrics
        story.append(Paragraph("Performance Overview", heading_style))
        
        if 'performance_distribution' in stats:
            perf_data = [['Performance Rating', 'Count', 'Percentage']]
            total_successful = stats.get('successful_analysis', 1)
            for rating, count in stats['performance_distribution'].items():
                percentage = (count / total_successful) * 100
                perf_data.append([rating.upper(), str(count), f"{percentage:.1f}%"])
            
            perf_table = Table(perf_data)
            perf_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(perf_table)
            story.append(Spacer(1, 20))
        
        # Lead Time Analysis
        story.append(Paragraph("Lead Time Analysis", heading_style))
        
        if 'lead_time_by_issue_type' in stats:
            lead_time_data = [['Issue Type', 'Avg Lead Time (min)', 'Sample Size']]
            for issue_type, lt_stats in stats['lead_time_by_issue_type'].items():
                lead_time_data.append([
                    issue_type.title(),
                    f"{lt_stats['avg_lead_time']:.2f}",
                    str(lt_stats['count'])
                ])
            
            lead_time_table = Table(lead_time_data)
            lead_time_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(lead_time_table)
        
        # Build PDF
        doc.build(story)
        print(f"✅ PDF report generated: {filename}")
        return filename
'''
    
    with open('pdf_report_generator.py', 'w', encoding='utf-8') as f:
        f.write(pdf_code)
    
    print("✅ Deployment package created")
    print("💡 Install dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    # Run main analysis
    results, stats = main()
    
    # Deploy dashboard
    deploy_streamlit()
    
    # Create deployment package
    create_deployment_package()