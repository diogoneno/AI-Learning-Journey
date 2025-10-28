# 🚀 Day 4: Advanced Prompt Engineering

## 📌 Learning Objectives
- Fine-tune AI responses with **Temperature, Top-p, Frequency, and Presence Penalty**.
- Use **Role-based prompting** to enhance specificity.
- Experiment with different **OpenAI API parameters**.

---

## 🔥 Advanced Prompt Tuning
### ✅ **Temperature (0 to 1)**
- `0`: More deterministic responses.
- `1`: More diverse and creative.

### ✅ **Top-p (Nucleus Sampling)**
- `1.0`: Considers all tokens.
- `0.9`: Uses the top 90% most likely words.

### ✅ **Frequency & Presence Penalty**
- **Frequency**: Reduces repetition.
- **Presence**: Encourages diverse responses.

---

## 📝 Python Script Overview
- **File:** `advanced_prompt_engineering.py`
- Uses OpenAI API to test **advanced prompt tuning parameters**.
- Implements **role-based prompting**.

### 🔧 **Setup**
1. **Install dependencies**:
   ```bash
   pip install openai python-dotenv
