import openai
import os
from dotenv import load_dotenv

# Load API Key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def chat_with_ai():
    """Simple chatbot that remembers conversation history."""
    conversation_history = []
    
    print("🤖 AI Chatbot (Type 'exit' to quit)")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        # Append user message to conversation history
        conversation_history.append({"role": "user", "content": user_input})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation_history
        )

        bot_reply = response["choices"][0]["message"]["content"]
        print(f"Chatbot: {bot_reply}")

        # Append AI response to conversation history
        conversation_history.append({"role": "assistant", "content": bot_reply})

if __name__ == "__main__":
    chat_with_ai()
