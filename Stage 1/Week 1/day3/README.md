# 🎯 Day 3: Prompt Engineering Basics

## 📌 Learning Objectives
- Understand **Prompt Engineering** and its importance.
- Learn **Zero-shot, Few-shot, and Chain-of-Thought prompting**.
- Experiment with **OpenAI API** for effective prompt crafting.

---

## 🚀 Prompting Techniques
### ✅ **Zero-shot Prompting**
- No examples, just instructions.
- Example:
Translate "Goodbye" to French.

### ✅ **Few-shot Prompting**
- Provides examples to guide AI response.
- Example:

English: Hello → French: Bonjour English: Thank you → French: Merci English: Goodbye → French:


### ✅ **Chain-of-Thought Prompting**
- Encourages AI to reason step-by-step.
- Example:
Q: If I have 10 apples and eat 3, how many are left? Think step by step before answering.

---

## 📝 Python Script Overview
- **File:** `prompt_engineering.py`
- Uses **OpenAI API** to test different prompting techniques.
- Compares responses for **Zero-shot, Few-shot, and Chain-of-Thought**.

### 🔧 **Setup**
1. **Install dependencies**:
 ```bash
 pip install openai python-dotenv
 
 
 Create a .env file with:
OPENAI_API_KEY=your_api_key_here


Run the script:

python prompt_engineering.py
