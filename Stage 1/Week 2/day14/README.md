# 📄 Day 14: Mini Project - AI Document Summarizer  

## 📌 Learning Objectives  
- Build an **AI-powered document processing tool**.  
- Extract **text from PDFs** and generate **summaries**.  
- Implement a **simple Gradio web UI** for easy interaction.  

---

## 🚀 Why Build an AI Document Summarizer?  
✅ **Quickly summarize long documents** like research papers, reports, or contracts.  
✅ **Automate document processing** with AI-generated summaries.  
✅ **Make AI accessible via a user-friendly web interface.**  

---

## 📝 Python Script Overview  
- **File:** `pdf_summarizer_app.py`  
- Uses **PyPDF2** to extract text from PDFs.  
- Generates **AI-powered summaries** using a local LLM (via LM Studio).  
- Provides a **Gradio web UI** for document upload & summarization.  

### 🔧 **Setup**  
1. **Install dependencies**:  
   ```bash
   pip install PyPDF2 gradio requests langchain
