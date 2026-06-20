# AI Assistant Chatbot

A multimodal AI assistant with **Customer Service**, **Medical Q&A (MedQuAD)**, and **arXiv ML Expert** capabilities. Built with LangChain, Gemini, and FAISS — featuring RAG, Vision, multilingual support, and sentiment analysis.

## Features

### 💬 Customer Service Chatbot
- **RAG pipeline** — retrieves relevant FAQs from a FAISS vector store built from CSV/URLs/images
- **Image analysis** — upload product screenshots or documents; Gemini Vision extracts text and context
- **Ambiguity detection** — automatically detects vague questions and asks for clarification
- **Sentiment analysis** — detects customer emotion and flags escalation-worthy interactions
- **Multilingual support** — detects and responds in 10 languages (English, Hindi, Arabic, French, Spanish, German, Chinese, Japanese, Portuguese, Russian)
- **Auto-update scheduler** — periodically refreshes the knowledge base from web sources

### 🏥 Medical Q&A (MedQuAD)
- **47,457 NIH-sourced Q&A pairs** from 12 NIH websites
- **Medical NER** — extracts diseases, symptoms, treatments, drugs, and tests from queries
- **FAISS-based retrieval** over MedQuAD content
- **Disclaimer enforcement** — every response includes a medical disclaimer

### 🔬 arXiv ML Expert
- **Semantic paper search** over ~200K ML papers (cs.LG, stat.ML, cs.AI, cs.CV, cs.CL, cs.NE)
- **ML entity extraction** — detects methods, tasks, datasets, and metrics from queries
- **Paper summarization** — auto-generates plain-English summaries via Gemini
- **Concept explainer** — explains ML concepts with key points, analogies, and related terms
- **Concept co-occurrence graph** — interactive visualization of topic relationships
- **Multi-turn expert chat** with arXiv-grounded answers and citations

## Architecture

```
User Input (text/image)
        │
        ▼
┌───────────────────────────────┐
│  1. Language Detection         │  langdetect (offline)
│  2. Translation → English      │  deep-translator (if needed)
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│  3. Combined Classifier        │  Gemini: ambiguity + sentiment
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│  4. Image Analysis (if image)  │  Gemini Vision
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│  5. KB Retrieval (FAISS)       │  HuggingFace Embeddings
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│  6. Contextual Prompt Builder  │  history + vision + sentiment + lang
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│  7. LLM Generation (Gemini)    │  Structured JSON response
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│  8. Response Validation        │  Self-critique pass
└───────────────┬───────────────┘
                ▼
         Final Answer
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| UI | Streamlit |
| LLM | Google Gemini 1.5 Flash |
| Vector Store | FAISS (CPU) |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| RAG Framework | LangChain |
| Language Detection | langdetect (offline) |
| Translation | deep-translator (Google Translate wrapper) |
| Image Analysis | Gemini Vision API |
| Medical Dataset | MedQuAD (47,457 NIH Q&A pairs) |
| arXiv Dataset | Kaggle arXiv Metadata Snapshot (~1.7M papers) |
| Graph Visualization | pyvis |
| Task Scheduling | APScheduler |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-assistant-chatbot.git
cd ai-assistant-chatbot

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get a free API key at [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

## Usage

```bash
streamlit run src/main.py
```

### First-time setup

1. **Customer Service KB** — Go to *Knowledge Base Manager → Customer Service KB → Build KB from CSV*
2. **Medical Index** — Clone MedQuAD, set the path, and click *Build Medical Index*
3. **arXiv Index** — Download the Kaggle arXiv dataset, set the path, and click *Build arXiv Index*

### Datasets required

| Dataset | Source | Path |
|---------|--------|------|
| Customer FAQ CSV | Your own dataset | `src/dataset.csv` |
| MedQuAD | [GitHub](https://github.com/abachaa/MedQuAD) (CC BY 4.0) | `src/MedQuAD/` |
| arXiv Metadata | [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv) (CC0) | `src/arxiv-metadata-oai-snapshot.json` |

## Project Structure

```
├── src/
│   ├── main.py                 # Streamlit app entry point
│   ├── langchain_helper.py     # Core RAG pipeline (multimodal + multilingual)
│   ├── ingestion.py            # Knowledge base builder (CSV, web, images)
│   ├── medical_helper.py       # Medical Q&A with MedQuAD + NER
│   ├── arxiv_helper.py         # arXiv expert: search, summarization, concepts
│   ├── arxiv_ui.py             # arXiv tab UI components
│   ├── faiss_index/            # Customer service FAISS index
│   ├── medical_faiss_index/    # Medical FAISS index
│   └── arxiv_faiss_index/      # arXiv FAISS index
├── .env                        # API key (not committed)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Key Design Decisions

- **Combined classification** — Ambiguity + sentiment in a single Gemini call to reduce API calls on free-tier keys
- **Retry with exponential backoff** — Handles transient 429/503 errors from the free Gemini API
- **Offline language detection** — langdetect runs locally; zero API cost for multilingual support
- **Separate FAISS indexes** — Each domain (CS, medical, arXiv) has its own vector store for focused retrieval
- **Rule-based NER** — Medical and ML entity extraction uses keyword matching (fast, no extra API calls)

## Results & Metrics

- **Confidence tracking** — Per-response confidence scores averaged across the session
- **Sentiment distribution** — Positive/neutral/negative counts displayed in the sidebar
- **Escalation alerts** — High-intensity negative sentiment triggers a human hand-off warning
- **Validation pass** — Every answer runs through a self-critique step to catch hallucinations

## Model Comparison

| Model | Free Quota | Use Case |
|-------|-----------|----------|
| Gemini 1.5 Flash | 1,500 req/day | Current default — balances speed, quality, and quota |
| Gemini 2.0 Flash | 1,500 req/day | Alternative — similar quota, slightly different output style |
| Gemini 2.5 Flash | 20 req/day | Higher quality, very limited free quota |

## License

This project uses the following third-party datasets:
- **MedQuAD** — Creative Commons Attribution 4.0 (CC BY)
- **arXiv Metadata** — Creative Commons CC0

The source code is available for educational and reference purposes.
