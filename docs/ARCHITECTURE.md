# Portfolio Architecture

```mermaid
flowchart LR
    A[AI Engineering Portfolio] --> B[ML Foundations]
    A --> C[Deep Learning]
    A --> D[Language Models]
    A --> E[Applied GenAI]

    B --> B1[scikit-learn Pipeline]
    C --> C1[TensorFlow / Keras]
    C --> C2[PyTorch]
    D --> D1[Transformer NLP]
    D --> D2[LLM Fine-Tuning]
    E --> E1[RAG Retrieval]
    E --> E2[Agentic Workflow]

    B1 --> Q[Reproducible Evaluation]
    C1 --> Q
    C2 --> Q
    D1 --> Q
    D2 --> Q
    E1 --> Q
    E2 --> Q
```

## Design Principles
The repository separates classical ML, deep learning, language-model engineering and applied GenAI into focused demonstrations. Generated models, private datasets and runtime artifacts are excluded from source control. Claims are limited to what the committed implementation or credential supports.
