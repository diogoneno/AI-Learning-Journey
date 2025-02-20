import requests
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# API Endpoint from LM Studio
API_URL = "http://localhost:1234/v1/completions"

# Define a Prompt Template
prompt = PromptTemplate(
    input_variables=["question"],
    template="You are an AI assistant. Answer concisely: {question}"
)

# Define a function to query LM Studio API
def get_llm_response(question):
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "prompt": prompt.format(question=question),
        "max_tokens": 100,
        "temperature": 0.7
    }
    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["text"]
    return "Error fetching response"

# Using LangChain's LLMChain
chain = LLMChain(llm=get_llm_response, prompt=prompt)

# Testing the pipeline
question = "What is the importance of AI in cybersecurity?"
print("AI Response:", chain.run(question))
