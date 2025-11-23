
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
