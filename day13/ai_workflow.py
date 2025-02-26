import PyPDF2
import requests
from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# API Endpoint from LM Studio
API_URL = "http://localhost:1234/v1/completions"

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

# Define a prompt template for summarization
summarization_prompt = PromptTemplate(
    input_variables=["document"],
    template="Summarize this document in key points:\n\n{document}"
)

# Define a prompt template for Q&A
qa_prompt = PromptTemplate(
    input_variables=["summary", "question"],
    template="Based on the following summary:\n{summary}\nAnswer the question: {question}"
)

# Function to get AI-generated text
def get_ai_response(prompt_text):
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "prompt": prompt_text,
        "max_tokens": 200,
        "temperature": 0.7
    }
    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["text"]
    return "Error fetching response"

# Define AI chains
summarization_chain = LLMChain(llm=get_ai_response, prompt=summarization_prompt)
qa_chain = LLMChain(llm=get_ai_response, prompt=qa_prompt)

# Combine into a Sequential Chain
chain = SequentialChain(
    chains=[summarization_chain, qa_chain],
    input_variables=["document", "question"],
    output_variables=["summary", "answer"]
)

# Run the pipeline
pdf_path = "sample.pdf"  # Change to your PDF file
document_text = extract_text_from_pdf(pdf_path)

print("📄 AI Summary:")
summary = get_ai_response(summarization_prompt.format(document=document_text[:2000]))  # Limit input
print(summary)

while True:
    question = input("\nAsk a question about the document (or type 'exit' to stop): ")
    if question.lower() == "exit":
        break
    answer = get_ai_response(qa_prompt.format(summary=summary, question=question))
    print("\nAI Answer:", answer)
