import openai
import streamlit as st
import os
import speech_recognition as sr
import pyttsx3
from dotenv import load_dotenv

# Load API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize conversation history
conversation_history = []

# Initialize text-to-speech engine
engine = pyttsx3.init()

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

def recognize_speech():
    """Converts speech input to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎙️ Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the speech."
    except sr.RequestError:
        return "Error: Speech recognition service is unavailable."

def text_to_speech(text):
    """Converts text to speech."""
    engine.say(text)
    engine.runAndWait()

# Streamlit UI
st.title("🤖 AI Chatbot with Voice Features")
st.write("Type or Speak to Chat with AI")

# User input
user_input = st.text_input("Your Message:")
if st.button("Send"):
    if user_input:
        response = chat_with_ai(user_input)
        st.write("AI:", response)
        text_to_speech(response)

# Speech recognition button
if st.button("🎙️ Speak"):
    speech_text = recognize_speech()
    st.text(f"You said: {speech_text}")
    if speech_text:
        response = chat_with_ai(speech_text)
        st.write("AI:", response)
        text_to_speech(response)
