import requests
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# API Endpoint from LM Studio
API_URL = "http://localhost:1234/v1/completions"

# Define a function for AI-based web search
def search_web(query):
    search_url = f"https://api.duckduckgo.com/?q={query}&format=json"
    response = requests.get(search_url)
    if response.status_code == 200:
        return response.json().get("AbstractText", "No relevant information found.")
    return "Error fetching search results."

# Define a function for AI-generated responses
def get_ai_response(prompt_text):
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "prompt": prompt_text,
        "max_tokens": 150,
        "temperature": 0.7
    }
    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["text"]
    return "Error fetching response."

# Create an AI agent with a tool for web search
tools = [
    Tool(
        name="WebSearch",
        func=search_web,
        description="Search the web for information"
    )
]

agent = initialize_agent(
    tools=tools,
    llm=get_ai_response,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Running the agent
print("🤖 AI Agent is ready (Type 'exit' to stop)")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        print("AI Agent: Goodbye!")
        break
    response = agent.run(user_input)
    print("\nAI Agent:", response)
