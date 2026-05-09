# 1. Essential Installs
!pip install -q gradio pdfminer.six scikit-learn pandas plotly

import gradio as gr
import pandas as pd
import plotly.express as px
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def process_recruitment(job_desc, resumes, min_score):
    # Error check for empty inputs
    if not job_desc or not resumes:
        return None, None, None

    results = []
    
    # Logic to handle one or multiple resumes correctly
    for file in resumes:
        # Extract text directly from the Gradio file object
        text = extract_text(file.name)
        
        # AI Matching logic
        vectorizer = TfidfVectorizer().fit_transform([job_desc, text])
        vectors = vectorizer.toarray()
        # Compare JD (index 0) with Resume (index 1)
        score = round(cosine_similarity([vectors[0]], [vectors[1]])[0][0] * 100, 2)
        
        results.append({
            "Candidate": file.name.split('/')[-1],
            "ATS Score (%)": score,
            "Status": "Shortlisted" if score >= min_score else "Rejected",
            "Notice Period": "30 Days",
            "Location": "Remote/India"
        })

    df = pd.DataFrame(results).sort_values(by="ATS Score (%)", ascending=False)
    
    # Analytics Chart
    fig = px.bar(df, x="Candidate", y="ATS Score (%)", color="Status", 
                 title="Hiring Pipeline Analysis", template="plotly_dark")
    
    # Export File
    df.to_csv("recruitment_report.csv", index=False)
    
    return df, fig, "recruitment_report.csv"

# Professional UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Track A: AI Recruitment Production Suite")
    
    with gr.Row():
        with gr.Column(scale=1):
            jd = gr.Textbox(label="Job Description", lines=8)
            res_input = gr.File(label="Upload Resumes", file_count="multiple", file_types=[".pdf"])
            threshold = gr.Slider(0, 100, value=50, label="Min Score Threshold")
            btn = gr.Button("Analyze Pipeline", variant="primary")
            
        with gr.Column(scale=2):
            table_out = gr.Dataframe(label="Dashboard")
            plot_out = gr.Plot(label="Analytics")
            file_out = gr.File(label="Download Report")

    btn.click(process_recruitment, [jd, res_input, threshold], [table_out, plot_out, file_out])

demo.launch(share=True, debug=True)
