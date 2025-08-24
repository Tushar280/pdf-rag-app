# PDF RAG App

A local app to ask questions over PDFs using vector search and LLMs.

## Tech Used
Python 3.12, Streamlit, pdfplumber, sentence-transformers, Chroma, ctransformers/OpenAI.

## How to Run

1. Install Python 3.12.
2. Create venv:  
   `python -m venv .venv && .venv\Scripts\activate`
3. `pip install --upgrade pip`
4. `pip install -r requirements.txt`
5. To use OpenAI:  
   `$env:LLM_BACKEND="openai"`  
   `$env:OPENAI_API_KEY="sk-..."`  
   `python -m streamlit run app.py`
6. To use local LLM:  
   Download a GGUF model (see models/README.txt)  
   `$env:LLM_BACKEND="ctransformers"`  
   `$env:LOCAL_LLM_MODEL_PATH="models\model-name.gguf"`  
   `$env:LOCAL_LLM_MODEL_TYPE="mistral"`  
   `python -m streamlit run app.py`

## Known Issues
- For Windows, use PowerShell/CMD for pip installs.
- Large PDFs may be slow; monitored via UI.

---

## Folder Structure

- `app.py`: Streamlit app.
- `cli.py`: Command-line utility.
- `rag/`: Core functionality modules.
- `sample_pdfs/`: Place test PDFs here.
- `models/`: Put GGUF model files here.

See code comments in each Python file for more.
