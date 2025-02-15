# 🤖 Day 5: Building an API-Based Chatbot  

## 📌 Learning Objectives  
- Implement a **simple chatbot** using OpenAI API.  
- Handle **conversation memory** for improved chat flow.  
- Deploy a chatbot in the **command-line interface (CLI)**.  

---

## 🚀 How It Works  
1️⃣ The chatbot takes **user input**.  
2️⃣ It sends **API requests** to OpenAI’s ChatGPT model.  
3️⃣ It **remembers past conversations** for context-awareness.  
4️⃣ The chatbot **responds** and continues the conversation.  

---

## 📝 Python Script Overview  
- **File:** `chatbot.py`  
- Uses OpenAI API to generate responses.  
- Tracks **conversation history** to maintain chat context.  

### 🔧 **Setup**  
1. **Install dependencies**:  
   ```bash
   pip install openai python-dotenv
