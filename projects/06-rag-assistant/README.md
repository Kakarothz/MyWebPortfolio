# 06 — RAG Knowledge Assistant

**Status: Retrieval implementation**

Local semantic-retrieval implementation using Sentence Transformers and FAISS. A text document is chunked, embedded and searched by cosine-style inner-product similarity over normalized vectors.

## Run
```bash
pip install -r requirements.txt
python rag.py document.txt "your question"
```

Generation is deliberately separated from retrieval. This makes retrieved evidence inspectable before adding an LLM and avoids presenting an untested generator as a completed grounded assistant.
