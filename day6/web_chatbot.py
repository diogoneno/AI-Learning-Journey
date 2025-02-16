import openai
import gradio as gr
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

conversation_history = []

def chat_with_ai(user_input):
    """Handles chat responses and maintains history."""
    global conversation_history

    conversation_history.append({"role": "user", "content": user_input})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation_history
    )

    bot_reply = response["choices"][0]["message"]["content"]
    conversation_history.append({"role": "assistant", "content": bot_reply})

    return bot_reply

# Gradio UI
with gr.Blocks() as chat_ui:
    gr.Markdown("## 🤖 AI Chatbot")
    chatbot = gr.Chatbot()
    message = gr.Textbox(placeholder="Type a message...")
    submit_btn = gr.Button("Send")

    def user_message(user_input, chat_history):
        reply = chat_with_ai(user_input)
        chat_history.append((user_input, reply))
        return chat_history

    submit_btn.click(user_message, inputs=[message, chatbot], outputs=[chatbot])

# Launch the app
if __name__ == "__main__":
    chat_ui.launch()