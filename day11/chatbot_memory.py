import requests
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# API Endpoint from LM Studio
API_URL = "http://localhost:1234/v1/completions"

# Initialize memory to track conversation history
memory = ConversationBufferMemory(memory_key="history")

# Define a Prompt Template with memory
prompt = PromptTemplate(
    input_variables=["history", "question"],
    template="Here is the conversation so far:\n{history}\nUser: {question}\nAI:"
)

# Function to interact with the local LLM
def get_llm_response(question):
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "prompt": prompt.format(history=memory.load_memory_variables({})["history"], question=question),
        "max_tokens": 100,
        "temperature": 0.7
    }
    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        reply = response.json()["choices"][0]["text"]
        memory.save_context({"question": question}, {"output": reply})
        return reply
    return "Error fetching response"

# Chat loop
print("🧠 AI Chatbot with Memory (Type 'exit' to stop)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    response = get_llm_response(user_input)
    print("Chatbot:", response)
