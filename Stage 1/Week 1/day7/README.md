# 🎙️ Day 7: AI Chatbot with Voice Input & Output  

## 📌 Learning Objectives  
- Extend the **chatbot** with **Speech-to-Text** (Voice Input).  
- Add **Text-to-Speech** (Voice Output).  
- Deploy an **interactive UI with Streamlit**.  

---

## 🚀 How It Works  
1️⃣ The chatbot takes **text or voice input**.  
2️⃣ It sends **API requests** to OpenAI’s ChatGPT model.  
3️⃣ The chatbot **responds via text and voice output**.  

---

## 📝 Python Script Overview  
- **File:** `enhanced_chatbot.py`  
- Uses **SpeechRecognition** for voice input.  
- Uses **pyttsx3** for text-to-speech.  
- Deploys an **interactive UI using Streamlit**.  

### 🔧 **Setup**  
1. **Install dependencies**:  
   ```bash
   pip install openai streamlit python-dotenv speechrecognition pyttsx3
