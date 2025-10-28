import PyPDF2
import requests
import gradio as gr
from langchain.prompts import PromptTemplate

# API Endpoint from LM Studio
API_URL = "http://localhost:1234/v1/completions"

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Define AI summarization prompt
prompt = PromptTemplate(
    input_variables=["document"],
    template="Summarize this document in key points:\n\n{document}"
)

# Function to generate a summary using AI
def summarize_pdf(pdf_file):
    document_text = extract_text_from_pdf(pdf_file)

    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "prompt": prompt.format(document=document_text[:2000]),  # Limit input size
        "max_tokens": 200,
        "temperature": 0.5
    }
    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["text"]
    return "Error fetching summary"

# Gradio UI
interface = gr.Interface(
    fn=summarize_pdf,
    inputs=gr.File(label="Upload a PDF"),
    outputs="text",
    title="📄 AI Document Summarizer",
    description="Upload a PDF and get a concise AI-generated summary."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
