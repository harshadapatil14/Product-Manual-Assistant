# Product Manual Assistant Using RAG

An intelligent assistant built with Retrieval-Augmented Generation (RAG) architecture that allows users to upload any product manual in PDF format and ask natural language questions about its contents. The system retrieves relevant context and generates precise answers using a local Large Language Model (LLaMA 3 via Ollama).

---

## Project Objective

To build a privacy-preserving, offline-ready assistant that:
- Extracts and processes product manuals
- Performs semantic search over extracted content
- Answers user queries using local LLMs
- Collects and analyzes user feedback

---

## Tech Stack

| Component        | Technology Used                         |
|------------------|------------------------------------------|
| UI Framework     | Streamlit                               |
| PDF Processing   | pdfplumber                              |
| Text Chunking    | LangChain’s RecursiveCharacterTextSplitter |
| Embedding Model  | Sentence Transformers (all-MiniLM-L6-v2)|
| Vector Database  | ChromaDB                                |
| Language Model   | LLaMA 3 (via [Ollama](https://ollama.com)) |
| Feedback Analysis| NLTK + VADER Sentiment Analysis         |

---

## System Workflow

1. **PDF Upload**  
   Users upload a product manual (PDF), which is parsed into raw text using `pdfplumber`.

2. **Text Chunking**  
   The text is split into overlapping chunks to preserve context using LangChain’s `RecursiveCharacterTextSplitter`.

3. **Embedding Generation**  
   Each chunk is converted into semantic embeddings using Sentence Transformers.

4. **Vector Storage**  
   Embeddings are stored in ChromaDB for efficient similarity search.

5. **Question Answering**  
   On user query:
   - Relevant chunks are retrieved from ChromaDB
   - Prompt is constructed using these chunks
   - Answer is generated using LLaMA 3 via Ollama (locally)

6. **Feedback Module**  
   Users can rate the response; feedback is analyzed using VADER and stored locally.

7. **Visualization**  
   Dashboard displays feedback distribution using Streamlit charts.

---

## Application UI

- **Upload Manual** tab: Upload and process PDF files
- **Ask Questions** tab: Chat with the assistant
- **Feedback** tab: Rate the response and view sentiment distribution
- **Team Info** tab: Display contributors and project metadata

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/harshadapatil14/Product-Manual-Assistant.git
cd Product-Manual-Assistant
````

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Ollama (LLaMA 3 must be installed)

```bash
ollama run llama3
```

### 5. Launch Streamlit App

```bash
streamlit run app.py
```

---

## Project Highlights

* Fully local & privacy-friendly — no API keys required
* Uses modern GenAI architecture (RAG)
* Modular Python codebase with clearly defined classes
* Extendable for multiple manuals or domains (legal, healthcare, etc.)
* Suitable for offline industrial documentation assistants

---

## Contributors

* **Harshada Patil**
  [GitHub Profile](https://github.com/harshadapatil14)
  
* **Pallavi Dudhalkar**
  [GitHub Profile](https://github.com/harshadapatil14)
  
---

## Future Enhancements

* Multi-manual support with project-specific memory
* Upload history and persistent vector storage
* Voice query support (Whisper integration)
* Admin dashboard to analyze usage trends
* Multilingual query handling

---
