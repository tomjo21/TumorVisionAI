from fpdf import FPDF
import datetime
import os

class MedicalReport(FPDF):
    def header(self):
        # We put title in the body for better single-page control
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(150)
        self.cell(0, 10, 'This report is AI-generated and should not be used as the sole basis for diagnosis.', 0, 0, 'C')

def generate_pdf_report_3d(metrics, img_path, original_filename, output_path):
    pdf = MedicalReport()
    pdf.add_page()
    
    # Header
    pdf.set_font('Arial', 'B', 22)
    pdf.set_text_color(34, 211, 238) # Cyan
    pdf.cell(0, 12, 'TumorVision AI | Medical Analysis', 0, 1, 'C')
    pdf.set_draw_color(34, 211, 238)
    pdf.line(10, 25, 200, 25)
    pdf.ln(10)
    
    # Patient Info Table
    pdf.set_fill_color(245, 247, 250)
    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(50)
    pdf.cell(30, 8, ' SCAN ID:', 1, 0, 'L', True)
    pdf.set_font('Arial', '', 10)
    pdf.cell(65, 8, f' {original_filename}', 1, 0, 'L')
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(30, 8, ' ANALYSIS DATE:', 1, 0, 'L', True)
    pdf.set_font('Arial', '', 10)
    pdf.cell(65, 8, f' {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}', 1, 1, 'L')
    
    pdf.ln(8)
    
    # Metrics
    pdf.set_font('Arial', 'B', 13)
    pdf.set_text_color(0)
    pdf.cell(0, 10, 'Volumetric Assessment', 0, 1)
    
    # Table Header
    pdf.set_font('Arial', 'B', 9)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(90, 7, "  Anatomical Layer / Region", 1, 0, 'L', True)
    pdf.cell(50, 7, "  Calculated Volume (cm3)", 1, 1, 'L', True)
    
    pdf.set_font('Arial', '', 9)
    regions = [
        ("Enhancing Tumor", metrics.get('enhancing_cm3', 0), (239, 68, 68)),
        ("Edema (Peritumoral)", metrics.get('edema_cm3', 0), (234, 179, 8)),
        ("Necrotic / Non-enhancing Core", metrics.get('necrotic_cm3', 0), (34, 211, 238)),
        ("Total Combined Tumor Burden", metrics.get('total_volume_cm3', 0), (0, 0, 0))
    ]
    
    for label, val, col in regions:
        pdf.set_text_color(0)
        pdf.cell(90, 7, f"  {label}", 1)
        pdf.set_text_color(*col)
        pdf.cell(50, 7, f"  {val:,.2f}", 1, 1) # Added comma formatting
        
    pdf.ln(8)
    
    # Visualization - Keep it compact
    pdf.set_font('Arial', 'B', 13)
    pdf.set_text_color(0)
    pdf.cell(0, 10, 'Diagnostic Image (Max Tumor Cross-Section)', 0, 1)
    
    if img_path and os.path.exists(img_path):
        # Image height constraint to avoid spill
        # Position centered
        pdf.image(img_path, x=60, y=pdf.get_y(), w=90)
        pdf.set_y(pdf.get_y() + 95) # Move cursor below image
    else:
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 10, "[Visualization generation skipped for this report]", 0, 1)

    pdf.ln(5)
    
    # Conclusion
    pdf.set_font('Arial', 'B', 11)
    pdf.set_text_color(0)
    pdf.cell(0, 7, 'System Summary:', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(60)
    pdf.multi_cell(0, 5, "Automated cross-sectional analysis identified significant abnormalities. "
                           "The volumetric data suggests localized tissue proliferation. Clinical correlation is mandatory.")

    pdf.output(output_path)
    return output_path

def generate_pdf_report_2d(prediction_label, confidence, img_path, original_filename, output_path):
    pdf = MedicalReport()
    pdf.add_page()
    
    # Header
    pdf.set_font('Arial', 'B', 22)
    pdf.set_text_color(34, 211, 238)
    pdf.cell(0, 12, 'TumorVision AI | Screening Result', 0, 1, 'C')
    pdf.set_draw_color(34, 211, 238)
    pdf.line(10, 25, 200, 25)
    pdf.ln(10)
    
    # Patient Info Table
    pdf.set_fill_color(245, 247, 250)
    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(50)
    pdf.cell(30, 8, ' IMAGE ID:', 1, 0, 'L', True)
    pdf.set_font('Arial', '', 10)
    pdf.cell(65, 8, f' {original_filename}', 1, 0, 'L')
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(30, 8, ' ANALYSIS DATE:', 1, 0, 'L', True)
    pdf.set_font('Arial', '', 10)
    pdf.cell(65, 8, f' {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}', 1, 1, 'L')
    
    pdf.ln(10)
    
    # Findings
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(0)
    pdf.cell(0, 10, 'Screening Findings', 0, 1)
    pdf.ln(2)
    
    # Result Box
    pdf.set_fill_color(250, 250, 250)
    pdf.rect(10, pdf.get_y(), 190, 20, 'F')
    
    pdf.set_xy(15, pdf.get_y() + 5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(45, 10, "PREDICTED TYPE:", 0, 0)
    
    if "no tumor" in prediction_label.lower():
         pdf.set_text_color(16, 185, 129)
    else:
         pdf.set_text_color(239, 68, 68)
         
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(60, 10, prediction_label.upper(), 0, 0)
    
    pdf.set_text_color(0)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(35, 10, "CONFIDENCE:", 0, 0)
    pdf.set_font('Arial', '', 12)
    pdf.cell(30, 10, confidence, 0, 1)
    
    pdf.set_xy(10, pdf.get_y() + 10)
    
    # Visualization
    if img_path and os.path.exists(img_path):
        pdf.set_font('Arial', 'B', 13)
        pdf.cell(0, 10, 'Reference Image', 0, 1)
        pdf.ln(2)
        pdf.image(img_path, x=60, y=pdf.get_y(), w=90)
        pdf.set_y(pdf.get_y() + 95)
        
    pdf.ln(10)
    
    # Interpretation
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, 'Clinical Interpretation:', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(60)
    
    if "no tumor" in prediction_label.lower():
        msg = "No obvious intracranial masses were identified by the AI system. Ensure regular checkups."
    else:
        msg = f"The AI system has identified features strongly suggestive of {prediction_label}. Radiological confirmation is required."
        
    pdf.multi_cell(0, 6, msg)
    
    pdf.output(output_path)
    return output_path
