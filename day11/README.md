# 🧠 Day 11: Memory & Context Management in AI  

## 📌 Learning Objectives  
- Understand **AI conversation memory**.  
- Implement **LangChain Memory** to track interactions.  
- Build an **AI chatbot with memory** using a local LLM.  

---

## 🚀 Why is AI Memory Important?  
Most AI models treat each query **independently**.  
✅ With **memory**, AI can recall previous messages.  
✅ Enables **context-aware responses** in chatbots.  

---

## 📝 Python Script Overview  
- **File:** `chatbot_memory.py`  
- Uses **LangChain Memory** for conversation tracking.  
- Connects to **a local model in LM Studio**.  

### 🔧 **Setup**  
1. **Install dependencies**:  
   ```bash
   pip install langchain openai requests
