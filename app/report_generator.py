import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle


REPORTS_DIR = os.path.join('app', 'static', 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)


def generate_report(patient_id, prediction_label, confidence_score, recommendations, image_path=None):
    """Generate a PDF report for a retinal image prediction result."""
    report_name = f'report_{patient_id}_{datetime.now().strftime("%Y%m%d%H%M%S")}.pdf'
    file_path = os.path.join(REPORTS_DIR, report_name)

    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph('RetinaHeartPredict Medical Report', styles['Title']))
    story.append(Paragraph('Clinical AI Retinal Image Assessment', styles['Heading2']))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f'Patient ID: {patient_id}', styles['BodyText']))
    story.append(Paragraph(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', styles['BodyText']))
    story.append(Paragraph(f'Prediction Result: {prediction_label}', styles['BodyText']))
    story.append(Paragraph(f'Confidence Score: {confidence_score:.2%}', styles['BodyText']))
    story.append(Spacer(1, 12))

    if image_path and os.path.exists(image_path):
        story.append(Paragraph('Retinal Image', styles['Heading3']))
        story.append(Image(image_path, width=180, height=180))
        story.append(Spacer(1, 12))

    recommendations_list = recommendations if isinstance(recommendations, list) else [recommendations]
    table_data = [['Recommendation']] + [[item] for item in recommendations_list]
    table = Table(table_data, colWidths=[450])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1d4ed8')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#111827')),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#0f172a'), colors.HexColor('#111827')]),
    ]))
    story.append(table)

    doc.build(story)
    return file_path
