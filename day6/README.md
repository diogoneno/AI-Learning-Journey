# 🌐 Day 6: Simple Chatbot Web App  

## 📌 Learning Objectives  
- Convert the CLI chatbot into a **web-based chatbot**.  
- Use **Gradio** to build a user-friendly interface.  
- Maintain **conversation history** for better interactions.  

---

## 🚀 How It Works  
1️⃣ The chatbot takes **user input**.  
2️⃣ It sends **API requests** to OpenAI’s ChatGPT model.  
3️⃣ It **remembers past conversations**.  
4️⃣ The chatbot **displays responses** in a web interface.  

---

## 📝 Python Script Overview  
- **File:** `web_chatbot.py`  
- Uses **Gradio** for a simple web-based UI.  
- Tracks **conversation history** to maintain context.  

### 🔧 **Setup**  
1. **Install dependencies**:  
   ```bash
   pip install openai gradio python-dotenv