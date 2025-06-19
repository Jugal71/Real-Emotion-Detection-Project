import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from src.config.model_config import ModelConfig

class MedicalReportGenerator:
    def __init__(self):
        self.session_data = []
        self.session_start = datetime.now()

    def add_detection(self, frame_number, results):
        """Add detection results to session data."""
        timestamp = datetime.now()
        for result in results:
            self.session_data.append({
                'timestamp': timestamp,
                'frame_number': frame_number,
                'emotion': result['emotion'],
                'confidence': result['confidence'],
                'all_emotions': result['all_emotions']
            })

    def get_emotion_summary(self):
        """Generate summary statistics for emotions."""
        if not self.session_data:
            return None

        df = pd.DataFrame(self.session_data)

        summary = {
            'session_duration': str(datetime.now() - self.session_start),
            'total_frames': len(df['frame_number'].unique()),
            'emotion_distribution': df['emotion'].value_counts().to_dict(),
            'average_confidence': df['confidence'].mean(),
            'emotion_trends': {}
        }

        # Calculate emotion trends
        for emotion in ModelConfig.EMOTION_LABELS.values():
            emotion_data = df[df['emotion'] == emotion]
            if not emotion_data.empty:
                summary['emotion_trends'][emotion] = {
                    'count': len(emotion_data),
                    'avg_confidence': emotion_data['confidence'].mean(),
                    'max_confidence': emotion_data['confidence'].max(),
                    'min_confidence': emotion_data['confidence'].min()
                }

        return summary

    def generate_report(self, patient_id=None, notes=None):
        """Generate a medical report with visualizations."""
        summary = self.get_emotion_summary()
        if not summary:
            return "No data available for report generation."

        # Create report directory
        report_dir = os.path.join(ModelConfig.LOG_DIR, 'medical_reports')
        os.makedirs(report_dir, exist_ok=True)

        # Generate report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_id = f"{patient_id}_{timestamp}" if patient_id else timestamp
        report_path = os.path.join(report_dir, f"medical_report_{report_id}")

        # Generate visualizations
        self._generate_visualizations(report_path)

        # Create HTML report
        html_content = self._generate_html_report(summary, patient_id, notes)

        with open(f"{report_path}.html", 'w') as f:
            f.write(html_content)

        return f"{report_path}.html"

    def _generate_visualizations(self, report_path):
        """Generate visualization plots for the report."""
        df = pd.DataFrame(self.session_data)

        # Emotion distribution pie chart
        plt.figure(figsize=(10, 6))
        df['emotion'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('Emotion Distribution')
        plt.savefig(f"{report_path}_emotion_dist.png")
        plt.close()

        # Confidence trends
        plt.figure(figsize=(12, 6))
        for emotion in ModelConfig.EMOTION_LABELS.values():
            emotion_data = df[df['emotion'] == emotion]
            if not emotion_data.empty:
                plt.plot(emotion_data['frame_number'],
                        emotion_data['confidence'],
                        label=emotion)
        plt.title('Emotion Confidence Trends')
        plt.xlabel('Frame Number')
        plt.ylabel('Confidence')
        plt.legend()
        plt.savefig(f"{report_path}_confidence_trends.png")
        plt.close()

    def _generate_html_report(self, summary, patient_id, notes):
        """Generate HTML report content."""
        html_template = f"""
        <html>
        <head>
            <title>Medical Emotion Detection Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; }}
                .content {{ margin-top: 20px; }}
                .visualization {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f8f9fa; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Medical Emotion Detection Report</h1>
                <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                {f'<p>Patient ID: {patient_id}</p>' if patient_id else ''}
            </div>

            <div class="content">
                <h2>Session Summary</h2>
                <table>
                    <tr><th>Session Duration</th><td>{summary['session_duration']}</td></tr>
                    <tr><th>Total Frames</th><td>{summary['total_frames']}</td></tr>
                    <tr><th>Average Confidence</th><td>{summary['average_confidence']:.2f}</td></tr>
                </table>

                <h2>Emotion Analysis</h2>
                <table>
                    <tr>
                        <th>Emotion</th>
                        <th>Count</th>
                        <th>Avg Confidence</th>
                        <th>Max Confidence</th>
                        <th>Min Confidence</th>
                    </tr>
                    {''.join(f'''
                    <tr>
                        <td>{emotion}</td>
                        <td>{data['count']}</td>
                        <td>{data['avg_confidence']:.2f}</td>
                        <td>{data['max_confidence']:.2f}</td>
                        <td>{data['min_confidence']:.2f}</td>
                    </tr>
                    ''' for emotion, data in summary['emotion_trends'].items())}
                </table>

                <div class="visualization">
                    <h2>Visualizations</h2>
                    <img src="medical_report_{patient_id if patient_id else datetime.now().strftime('%Y%m%d_%H%M%S')}_emotion_dist.png"
                         alt="Emotion Distribution">
                    <img src="medical_report_{patient_id if patient_id else datetime.now().strftime('%Y%m%d_%H%M%S')}_confidence_trends.png"
                         alt="Confidence Trends">
                </div>

                {f'''
                <div class="notes">
                    <h2>Clinical Notes</h2>
                    <p>{notes}</p>
                </div>
                ''' if notes else ''}
            </div>
        </body>
        </html>
        """
        return html_template
