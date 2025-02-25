import PyPDF2
import requests
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# API Endpoint from LM Studio
API_URL = "http://localhost:1234/v1/completions"

# Define a function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

# Define a prompt template for AI summarization
prompt = PromptTemplate(
    input_variables=["document"],
    template="Summarize the following document in key points:\n\n{document}"
)

# Function to get AI summary
def get_summary(text):
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "prompt": prompt.format(document=text[:2000]),  # Limit input size
        "max_tokens": 150,
        "temperature": 0.5
    }
    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["text"]
    return "Error fetching summary"

# Run the pipeline
pdf_path = "sample.pdf"  # Change this to your PDF file
document_text = extract_text_from_pdf(pdf_path)
summary = get_summary(document_text)

print("📄 Document Summary:\n", summary)
