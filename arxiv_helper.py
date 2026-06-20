import os
import re
import json
import logging
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv(r"c:\customer_service_chatbot_LLM\.env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths

ARXIV_FAISS_PATH  = r"c:\customer_service_chatbot_LLM\src\arxiv_faiss_index"
ARXIV_META_PATH   = r"c:\customer_service_chatbot_LLM\src\arxiv_meta.json"
ARXIV_JSON_PATH   = r"c:\customer_service_chatbot_LLM\src\arxiv-metadata-oai-snapshot.json"

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
genai.configure(api_key=GOOGLE_API_KEY)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1,
)

# Target categories

TARGET_CATEGORIES = {"cs.LG", "stat.ML", "cs.AI", "cs.NE", "cs.CV", "cs.CL"}

# ML / CS keyword lists for NLP entity extraction

_ML_METHODS = [
    "neural network", "deep learning", "machine learning", "reinforcement learning",
    "transformer", "attention mechanism", "bert", "gpt", "llm", "large language model",
    "convolutional neural network", "cnn", "recurrent neural network", "rnn", "lstm",
    "gru", "autoencoder", "variational autoencoder", "vae", "gan", "generative adversarial",
    "diffusion model", "graph neural network", "gnn", "random forest", "gradient boosting",
    "xgboost", "support vector machine", "svm", "logistic regression", "linear regression",
    "k-means", "dbscan", "principal component analysis", "pca", "t-sne", "umap",
    "federated learning", "transfer learning", "fine-tuning", "few-shot learning",
    "zero-shot learning", "meta-learning", "self-supervised learning", "contrastive learning",
    "knowledge distillation", "pruning", "quantization", "neural architecture search",
    "bayesian optimization", "hyperparameter tuning", "cross-validation",
]

_ML_TASKS = [
    "classification", "regression", "clustering", "object detection", "segmentation",
    "natural language processing", "nlp", "computer vision", "speech recognition",
    "machine translation", "text generation", "question answering", "summarization",
    "sentiment analysis", "named entity recognition", "ner", "information retrieval",
    "recommendation system", "anomaly detection", "time series forecasting",
    "image recognition", "image generation", "text-to-image", "multimodal",
]

_ML_DATASETS = [
    "imagenet", "cifar", "mnist", "coco", "glue", "superglue", "squad",
    "common crawl", "bookcorpus", "wikipedia", "openwebtext", "c4",
    "laion", "conceptual captions", "vqa", "clevr",
]

_ML_METRICS = [
    "accuracy", "precision", "recall", "f1 score", "auc", "roc", "bleu", "rouge",
    "perplexity", "top-1 accuracy", "top-5 accuracy", "mean average precision",
    "intersection over union", "iou", "word error rate", "wer",
]


# ML NLP entity extraction

def extract_ml_entities(text: str) -> dict:
    """
    Rule-based ML entity extractor for queries and paper text.

    Returns:
        {
            "methods":  [...],
            "tasks":    [...],
            "datasets": [...],
            "metrics":  [...],
        }
    """
    lower = text.lower()
    entities = {"methods": [], "tasks": [], "datasets": [], "metrics": []}

    def _find(keywords: list, category: str):
        for kw in keywords:
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, lower):
                entities[category].append(kw)

    _find(_ML_METHODS,  "methods")
    _find(_ML_TASKS,    "tasks")
    _find(_ML_DATASETS, "datasets")
    _find(_ML_METRICS,  "metrics")

    for k in entities:
        entities[k] = list(dict.fromkeys(entities[k]))

    return entities


def format_ml_entities_for_display(entities: dict) -> str:
    """Convert ML entity dict to a readable markdown string for the Streamlit UI."""
    icons = {
        "methods":  "🔧 **Methods / Models**",
        "tasks":    "🎯 **Tasks**",
        "datasets": "📦 **Datasets**",
        "metrics":  "📏 **Metrics**",
    }
    lines = []
    for category, label in icons.items():
        items = entities.get(category, [])
        if items:
            lines.append(f"{label}: {', '.join(items)}")
    return "\n".join(lines) if lines else "No specific ML entities detected."


# arXiv dataset loader

def _paper_matches_target(paper: dict) -> bool:
    """Return True if the paper belongs to a target ML/AI category."""
    cats = paper.get("categories", "")
    return any(tc in cats.split() for tc in TARGET_CATEGORIES)


def load_arxiv_dataset(
    json_path: str = ARXIV_JSON_PATH,
    max_papers: Optional[int] = 5000,
) -> list:
    """
    Stream-parse the arXiv JSON snapshot and load ML/AI papers.

    The Kaggle snapshot is a JSONL file (one JSON object per line).
    Each paper becomes one Document: title + abstract + metadata.

    Args:
        json_path:  Path to arxiv-metadata-oai-snapshot.json
        max_papers: Cap the number of papers loaded (None = all matching papers)

    Returns:
        List of LangChain Documents
    """
    if not os.path.exists(json_path):
        logger.error(f"❌ arXiv JSON not found at: {json_path}")
        return []

    logger.info(f"📂 Loading arXiv papers from {json_path}…")
    docs = []
    total_scanned = 0

    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_scanned += 1

            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not _paper_matches_target(paper):
                continue

            arxiv_id   = paper.get("id", "unknown")
            title      = paper.get("title", "").replace("\n", " ").strip()
            abstract   = paper.get("abstract", "").replace("\n", " ").strip()
            authors    = paper.get("authors", "")
            categories = paper.get("categories", "")
            year       = paper.get("update_date", "")[:4]

            if not title or not abstract:
                continue

            page_content = f"Title: {title}\n\nAbstract: {abstract}"

            doc = Document(
                page_content=page_content,
                metadata={
                    "arxiv_id":   arxiv_id,
                    "title":      title,
                    "authors":    authors,
                    "categories": categories,
                    "year":       year,
                    "type":       "arxiv_paper",
                    "source":     f"arXiv:{arxiv_id}",
                },
            )
            docs.append(doc)

            if max_papers and len(docs) >= max_papers:
                break

            if len(docs) % 500 == 0:
                logger.info(f"  Loaded {len(docs)} ML papers (scanned {total_scanned} total)…")

    logger.info(
        f"✅ arXiv load complete — {len(docs)} ML papers loaded "
        f"(scanned {total_scanned} total records)"
    )
    return docs


# FAISS index management

def build_arxiv_index(
    json_path: str = ARXIV_JSON_PATH,
    max_papers: Optional[int] = 5000,
    chunk_size: int = 800,
    chunk_overlap: int = 80,
) -> int:
    """
    Load arXiv ML papers, chunk, embed, and persist a dedicated FAISS index.

    Returns:
        Number of chunks indexed
    """
    logger.info("🏗️  Building arXiv FAISS index…")
    docs = load_arxiv_dataset(json_path, max_papers=max_papers)

    if not docs:
        logger.error("❌ No papers loaded. Check the json_path.")
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"✂️  Split into {len(chunks)} chunks")

    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(ARXIV_FAISS_PATH)

    meta = {
        "total_chunks": len(chunks),
        "total_papers": len(docs),
        "json_path": json_path,
        "max_papers": max_papers,
    }
    with open(ARXIV_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"💾 arXiv index saved to {ARXIV_FAISS_PATH}")
    return len(chunks)


def load_arxiv_vectordb():
    """Load the persisted arXiv FAISS index. Returns None if not built yet."""
    if not os.path.exists(ARXIV_FAISS_PATH):
        logger.warning("⚠️ arXiv FAISS index not found. Build it first via build_arxiv_index().")
        return None
    try:
        return FAISS.load_local(
            ARXIV_FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        logger.error(f"❌ Failed to load arXiv index: {e}")
        return None


def retrieve_arxiv_docs(question: str, k: int = 5) -> list:
    """
    Retrieve the top-k most relevant arXiv papers for a query.
    Returns empty list if index not built.
    """
    vectordb = load_arxiv_vectordb()
    if vectordb is None:
        return []
    try:
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
        docs = retriever.invoke(question)
        logger.info(f"🔍 arXiv retrieval — {len(docs)} docs for: '{question[:60]}'")
        return docs
    except Exception as e:
        logger.error(f"❌ arXiv retrieval failed: {e}")
        return []


# Paper summariser

def summarise_paper(title: str, abstract: str) -> str:
    """
    Generate a concise, plain-English summary of a paper using Gemini.
    Returns a 3-5 sentence summary.
    """
    prompt = f"""You are an expert ML researcher. Summarise the following paper in 3-5 clear sentences
for someone who understands ML but may not know this specific paper.
Focus on: what problem it solves, what approach it takes, and what the key result is.

Title: {title}

Abstract: {abstract}

Write only the summary — no labels, no bullet points, no extra text."""
    try:
        resp = llm.invoke(prompt)
        return resp.content.strip()
    except Exception as e:
        logger.error(f"❌ Paper summarisation failed: {e}")
        return "Summary unavailable. Please check the abstract directly."


# Concept explainer

def explain_concept(concept: str, retrieved_context: str = "") -> dict:
    """
    Explain an ML/CS concept in plain English using Gemini,
    optionally grounded in retrieved arXiv paper context.

    Returns:
        {
            "explanation":   str,
            "key_points":    list[str],
            "related_terms": list[str],
            "example":       str,
            "papers":        list[str],
        }
    """
    context_block = f"\n\nRelevant arXiv context:\n{retrieved_context[:1200]}" if retrieved_context else ""

    prompt = f"""You are an expert ML educator. Explain the following concept clearly and accurately.{context_block}

Concept to explain: "{concept}"

Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "explanation": "2-3 paragraph plain-English explanation suitable for an ML practitioner",
  "key_points": ["Key point 1", "Key point 2", "Key point 3", "Key point 4"],
  "related_terms": ["related concept 1", "related concept 2", "related concept 3"],
  "example": "One concrete, intuitive example or analogy to illustrate the concept",
  "papers": ["Influential paper 1 (author, year)", "Influential paper 2 (author, year)"]
}}"""

    try:
        resp = llm.invoke(prompt)
        raw = resp.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
        raise ValueError("No JSON found")
    except Exception as e:
        logger.error(f"❌ Concept explanation failed: {e}")
        return {
            "explanation": f"Unable to generate explanation for '{concept}'.",
            "key_points": [],
            "related_terms": [],
            "example": "",
            "papers": [],
        }


# Concept co-occurrence graph data

def build_concept_graph(papers: list, top_n: int = 15) -> dict:
    """
    Build a concept co-occurrence graph from a list of arXiv paper Documents.

    Returns:
        {
            "nodes": [{"id": "concept", "count": N}, ...],
            "edges": [{"source": "concept_a", "target": "concept_b", "weight": N}, ...]
        }
    """
    from collections import Counter

    concept_counts = Counter()
    co_occurrence  = Counter()

    for doc in papers:
        text = doc.page_content
        ents = extract_ml_entities(text)
        all_concepts = list(dict.fromkeys(
            ents["methods"] + ents["tasks"] + ents["datasets"] + ents["metrics"]
        ))
        for concept in all_concepts:
            concept_counts[concept] += 1
        for i in range(len(all_concepts)):
            for j in range(i + 1, len(all_concepts)):
                pair = tuple(sorted([all_concepts[i], all_concepts[j]]))
                co_occurrence[pair] += 1

    top_concepts = {c for c, _ in concept_counts.most_common(top_n)}

    nodes = [{"id": c, "count": concept_counts[c]} for c in top_concepts]
    edges = [
        {"source": a, "target": b, "weight": w}
        for (a, b), w in co_occurrence.items()
        if a in top_concepts and b in top_concepts and w >= 2
    ]

    return {"nodes": nodes, "edges": edges}


# arXiv expert response pipeline

def get_arxiv_response(
    question: str,
    conversation_history: list,
) -> dict:
    """
    Full arXiv expert pipeline.

    Steps:
      1. ML entity extraction (NER)
      2. arXiv FAISS retrieval
      3. Gemini-powered answer with citations
      4. Follow-up handling via conversation history

    Returns:
        {
            "answer":            str,
            "confidence":        float,
            "reasoning_steps":   list[str],
            "sources_used":      list[str],
            "context_reference": str,
            "ml_entities":       dict,
            "arxiv_sources":     list[dict],
            "paper_summaries":   list[dict],
        }
    """
    # Step 1: ML NER
    entities = extract_ml_entities(question)

    # Step 2: arXiv retrieval
    arxiv_docs = retrieve_arxiv_docs(question, k=5)

    arxiv_context = ""
    arxiv_sources = []
    paper_summaries = []

    for i, doc in enumerate(arxiv_docs):
        meta    = doc.metadata
        title   = meta.get("title", "Unknown")
        aid     = meta.get("arxiv_id", f"doc_{i}")
        year    = meta.get("year", "")
        cats    = meta.get("categories", "")
        authors = meta.get("authors", "")

        arxiv_context += (
            f"[arXiv:{aid} — {title} ({year})]\n"
            f"{doc.page_content.strip()}\n\n"
        )
        arxiv_sources.append({
            "arxiv_id":   aid,
            "title":      title,
            "authors":    authors,
            "year":       year,
            "categories": cats,
            "snippet":    doc.page_content[:200],
        })

        if i < 3:
            content   = doc.page_content
            abs_start = content.find("Abstract:")
            abstract  = content[abs_start + 9:].strip() if abs_start != -1 else content
            summary   = summarise_paper(title, abstract)
            paper_summaries.append({
                "arxiv_id": aid,
                "title":    title,
                "year":     year,
                "summary":  summary,
            })

    # Step 3: Build conversation history context
    history_text = ""
    for msg in conversation_history[-6:]:
        role = "User" if msg["role"] == "user" else "Expert"
        history_text += f"{role}: {msg['content']}\n"

    entity_summary = "; ".join(
        f"{k}: {', '.join(v)}" for k, v in entities.items() if v
    ) or "none detected"

    # Step 4: Gemini generation
    prompt = f"""You are a world-class ML researcher and educator with deep expertise in machine learning,
deep learning, and AI. You answer questions by synthesising insights from arXiv research papers.

══ CONVERSATION HISTORY ══
{history_text or "(First message)"}

══ DETECTED ML ENTITIES ══
{entity_summary}

══ RELEVANT arXiv PAPERS ══
{arxiv_context or "(No relevant papers found — answer from general ML knowledge.)"}

══ USER QUESTION ══
{question}

══ INSTRUCTIONS ══
Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "answer": "Comprehensive, expert answer. Cite specific papers using [arXiv:ID] notation. Explain concepts clearly. Handle follow-up questions by referencing prior conversation context.",
  "confidence": 0.0,
  "reasoning_steps": [
    "Step 1 — which entities / concepts are in the question",
    "Step 2 — which papers are most relevant and why",
    "Step 3 — how the answer synthesises the evidence"
  ],
  "sources_used": ["arXiv:XXXX.XXXXX — Title", "..."],
  "context_reference": "Which part of conversation history this relates to, if any"
}}

Set confidence: 0.85+ when papers directly answer the question, 0.6-0.85 for partial, <0.6 for general knowledge only.
"""

    parsed = {}
    try:
        resp = llm.invoke(prompt)
        raw  = resp.content.strip()
        raw  = re.sub(r"```json|```", "", raw).strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(raw[start:end])
        else:
            raise ValueError("No JSON found")
    except Exception as e:
        logger.error(f"❌ arXiv LLM call failed: {e}")
        parsed = {
            "answer": (
                "I encountered an error retrieving arXiv information. "
                "Please try rephrasing your question."
            ),
            "confidence": 0.3,
            "reasoning_steps": ["LLM call failed."],
            "sources_used": [],
            "context_reference": "",
        }

    return {
        "answer":            parsed.get("answer", ""),
        "confidence":        float(parsed.get("confidence", 0.6)),
        "reasoning_steps":   parsed.get("reasoning_steps", []),
        "sources_used":      parsed.get("sources_used", []),
        "context_reference": parsed.get("context_reference", ""),
        "ml_entities":       entities,
        "arxiv_sources":     arxiv_sources,
        "paper_summaries":   paper_summaries,
    }


if __name__ == "__main__":
    q = "What are the latest advances in transformer attention mechanisms?"
    ents = extract_ml_entities(q)
    print("ML Entities:", json.dumps(ents, indent=2))

    result = get_arxiv_response(q, conversation_history=[])
    print("\narXiv Expert Response:")
    print(json.dumps(result, indent=2, default=str))
