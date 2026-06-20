import os
import re
import json
import logging
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv(r"c:\customer_service_chatbot_LLM\.env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths

MEDICAL_FAISS_PATH  = r"c:\customer_service_chatbot_LLM\src\medical_faiss_index"
MEDQUAD_DATASET_DIR = r"c:\customer_service_chatbot_LLM\src\MedQuAD"
MEDICAL_META_PATH   = r"c:\customer_service_chatbot_LLM\src\medical_meta.json"

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
genai.configure(api_key=GOOGLE_API_KEY)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Medical entity keyword lists

_SYMPTOM_KEYWORDS = [
    "pain", "ache", "fever", "cough", "fatigue", "nausea", "vomiting",
    "diarrhea", "headache", "dizziness", "shortness of breath", "swelling",
    "rash", "itching", "bleeding", "weakness", "numbness", "tingling",
    "chest pain", "back pain", "sore throat", "runny nose", "chills",
    "night sweats", "weight loss", "weight gain", "loss of appetite",
    "insomnia", "anxiety", "depression", "confusion", "blurred vision",
    "hearing loss", "joint pain", "muscle pain", "abdominal pain",
]

_DISEASE_KEYWORDS = [
    "diabetes", "hypertension", "cancer", "asthma", "arthritis", "alzheimer",
    "parkinson", "epilepsy", "stroke", "heart disease", "pneumonia", "tuberculosis",
    "hiv", "aids", "hepatitis", "cirrhosis", "kidney disease", "lupus",
    "multiple sclerosis", "crohn", "colitis", "psoriasis", "eczema",
    "fibromyalgia", "anemia", "thyroid", "hypothyroidism", "hyperthyroidism",
    "celiac", "gerd", "ibs", "copd", "atrial fibrillation", "sepsis",
    "meningitis", "osteoporosis", "scoliosis", "migraine",
]

_TREATMENT_KEYWORDS = [
    "surgery", "chemotherapy", "radiation", "therapy", "treatment",
    "medication", "drug", "vaccine", "immunotherapy", "transplant",
    "dialysis", "rehabilitation", "physical therapy", "psychotherapy",
    "antibiotic", "antiviral", "anti-inflammatory", "analgesic",
    "insulin", "steroid", "hormone therapy", "blood transfusion",
    "biopsy", "endoscopy", "mri", "ct scan", "x-ray", "ultrasound",
]

_DRUG_KEYWORDS = [
    "aspirin", "ibuprofen", "acetaminophen", "paracetamol", "metformin",
    "lisinopril", "atorvastatin", "levothyroxine", "amlodipine", "omeprazole",
    "metoprolol", "albuterol", "gabapentin", "sertraline", "fluoxetine",
    "amoxicillin", "azithromycin", "ciprofloxacin", "prednisone",
    "warfarin", "clopidogrel", "losartan", "simvastatin", "hydrochlorothiazide",
]

_TEST_KEYWORDS = [
    "blood test", "urine test", "biopsy", "ct scan", "mri", "x-ray",
    "ultrasound", "ecg", "ekg", "endoscopy", "colonoscopy", "mammogram",
    "pap smear", "psa test", "glucose test", "cholesterol test",
    "cbc", "complete blood count", "metabolic panel", "thyroid test",
]


# Medical NER — rule-based entity recognition

def extract_medical_entities(text: str) -> dict:
    """
    Lightweight rule-based medical NER.

    Scans text for medical entities using curated keyword lists.
    Returns a dict of entity categories to lists of found terms.

    Args:
        text: Free-form user question or document text

    Returns:
        {
            "diseases":   [...],
            "symptoms":   [...],
            "treatments": [...],
            "drugs":      [...],
            "tests":      [...],
        }
    """
    lower = text.lower()
    entities = {
        "diseases":   [],
        "symptoms":   [],
        "treatments": [],
        "drugs":      [],
        "tests":      [],
    }

    def _find(keywords: list, category: str):
        for kw in keywords:
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, lower):
                entities[category].append(kw)

    _find(_DISEASE_KEYWORDS,   "diseases")
    _find(_SYMPTOM_KEYWORDS,   "symptoms")
    _find(_TREATMENT_KEYWORDS, "treatments")
    _find(_DRUG_KEYWORDS,      "drugs")
    _find(_TEST_KEYWORDS,      "tests")

    for k in entities:
        entities[k] = list(dict.fromkeys(entities[k]))

    logger.info(f"🏷️  NER — {sum(len(v) for v in entities.values())} entities found")
    return entities


def format_entities_for_display(entities: dict) -> str:
    """Convert NER dict to a readable markdown string for the Streamlit UI."""
    lines = []
    icons = {
        "diseases":   "🦠 **Diseases/Conditions**",
        "symptoms":   "🩺 **Symptoms**",
        "treatments": "💊 **Treatments/Procedures**",
        "drugs":      "💉 **Drugs/Medications**",
        "tests":      "🔬 **Tests/Diagnostics**",
    }
    for category, label in icons.items():
        items = entities.get(category, [])
        if items:
            lines.append(f"{label}: {', '.join(items)}")
    return "\n".join(lines) if lines else "No specific medical entities detected."


# MedQuAD loader — parse NIH XML files into LangChain Documents

def parse_medquad_xml(xml_path: str) -> list[Document]:
    """
    Parse a single MedQuAD XML file.

    MedQuAD XML structure (simplified):
    <Document>
      <Focus>disease/topic name</Focus>
      <QAPairs>
        <QAPair pid="1">
          <Question qtype="symptoms">...</Question>
          <Answer>...</Answer>
        </QAPair>
        ...
      </QAPairs>
    </Document>

    Each QA pair becomes one Document with rich metadata.
    """
    docs = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        focus_el = root.find(".//Focus")
        focus = focus_el.text.strip() if focus_el is not None and focus_el.text else "Unknown"

        source_name = Path(xml_path).parent.name

        for pair in root.findall(".//QAPair"):
            q_el = root.find(f".//QAPair[@pid='{pair.get('pid')}']/Question")
            a_el = root.find(f".//QAPair[@pid='{pair.get('pid')}']/Answer")

            q_el = pair.find("Question")
            a_el = pair.find("Answer")

            if q_el is None or a_el is None:
                continue
            if not q_el.text or not a_el.text:
                continue

            question = q_el.text.strip()
            answer   = a_el.text.strip()
            qtype    = q_el.get("qtype", "general")

            page_content = f"Question: {question}\nAnswer: {answer}"

            doc = Document(
                page_content=page_content,
                metadata={
                    "source":      source_name,
                    "file":        Path(xml_path).name,
                    "focus":       focus,
                    "qtype":       qtype,
                    "question":    question,
                    "answer":      answer,
                    "type":        "medquad",
                },
            )
            docs.append(doc)

    except Exception as e:
        logger.warning(f"⚠️ Failed to parse {xml_path}: {e}")

    return docs


def load_medquad_dataset(
    dataset_dir: str = MEDQUAD_DATASET_DIR,
    max_files: Optional[int] = None,
) -> list[Document]:
    """
    Walk the MedQuAD directory tree and parse all XML files.

    Args:
        dataset_dir: Root directory of the cloned MedQuAD repo
        max_files:   Cap the number of XML files parsed (useful for quick tests)

    Returns:
        List of Documents ready for embedding
    """
    xml_files = list(Path(dataset_dir).rglob("*.xml"))
    if max_files:
        xml_files = xml_files[:max_files]

    logger.info(f"📂 Found {len(xml_files)} MedQuAD XML files")

    all_docs = []
    for i, xml_path in enumerate(xml_files):
        docs = parse_medquad_xml(str(xml_path))
        all_docs.extend(docs)
        if (i + 1) % 50 == 0:
            logger.info(f"  Parsed {i+1}/{len(xml_files)} files — {len(all_docs)} docs so far")

    logger.info(f"✅ MedQuAD load complete — {len(all_docs)} Q&A pairs loaded")
    return all_docs


# FAISS index management

def build_medical_index(
    dataset_dir: str = MEDQUAD_DATASET_DIR,
    max_files: Optional[int] = None,
    chunk_size: int = 600,
    chunk_overlap: int = 60,
) -> int:
    """
    Load MedQuAD dataset, chunk, embed, and persist a dedicated FAISS index.

    Args:
        dataset_dir:  Path to the cloned MedQuAD repo
        max_files:    Limit XML files parsed (None = all ~900 files)
        chunk_size:   Characters per chunk
        chunk_overlap: Overlap between chunks

    Returns:
        Number of chunks indexed
    """
    logger.info("🏗️  Building medical FAISS index…")
    docs = load_medquad_dataset(dataset_dir, max_files=max_files)

    if not docs:
        logger.error("❌ No documents loaded. Check MEDQUAD_DATASET_DIR path.")
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"✂️  Split into {len(chunks)} chunks")

    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(MEDICAL_FAISS_PATH)

    meta = {
        "total_chunks": len(chunks),
        "total_qa_pairs": len(docs),
        "dataset_dir": dataset_dir,
        "max_files": max_files,
    }
    with open(MEDICAL_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"💾 Medical index saved to {MEDICAL_FAISS_PATH}")
    return len(chunks)


def load_medical_vectordb():
    """Load the persisted medical FAISS index. Returns None if not built yet."""
    if not os.path.exists(MEDICAL_FAISS_PATH):
        logger.warning("⚠️ Medical FAISS index not found. Build it first via build_medical_index().")
        return None
    try:
        return FAISS.load_local(
            MEDICAL_FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        logger.error(f"❌ Failed to load medical index: {e}")
        return None


def retrieve_medical_docs(question: str, k: int = 5) -> list[Document]:
    """
    Retrieve the top-k most relevant MedQuAD documents for a question.

    Args:
        question: User's medical question
        k:        Number of documents to retrieve

    Returns:
        List of Documents (may be empty if index not built)
    """
    vectordb = load_medical_vectordb()
    if vectordb is None:
        return []
    try:
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
        docs = retriever.invoke(question)
        logger.info(f"🔍 Medical retrieval — {len(docs)} docs for: '{question[:60]}…'")
        return docs
    except Exception as e:
        logger.error(f"❌ Medical retrieval failed: {e}")
        return []


# Medical response builder

def get_medical_response(
    question: str,
    conversation_history: list,
    image_bytes: Optional[bytes] = None,
) -> dict:
    """
    Medical-domain wrapper around the multimodal pipeline.

    Extra steps compared to get_multimodal_response:
      1. Medical NER on the question
      2. MedQuAD retrieval (separate medical index)
      3. Injects medical context + disclaimer into the response

    Args:
        question:             User's medical question
        conversation_history: Same format as get_multimodal_response
        image_bytes:          Optional image bytes

    Returns:
        Same dict as get_multimodal_response, plus:
          "medical_entities": dict from extract_medical_entities()
          "medquad_sources":  list of MedQuAD source metadata dicts
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,
    )

    # NER
    entities = extract_medical_entities(question)

    # MedQuAD retrieval
    medical_docs = retrieve_medical_docs(question, k=5)

    medquad_context = ""
    medquad_sources = []
    for i, doc in enumerate(medical_docs):
        meta  = doc.metadata
        focus = meta.get("focus", "Unknown")
        qtype = meta.get("qtype", "general")
        src   = meta.get("source", f"medquad_{i}")
        medquad_context += (
            f"[MedQuAD Source {i+1} — {focus} ({qtype}) from {src}]\n"
            f"{doc.page_content.strip()}\n\n"
        )
        medquad_sources.append({
            "focus":  focus,
            "qtype":  qtype,
            "source": src,
            "snippet": doc.page_content[:200],
        })

    # Conversation history context
    history_text = ""
    for msg in conversation_history[-6:]:
        role = "Patient" if msg["role"] == "user" else "MedBot"
        history_text += f"{role}: {msg['content']}\n"

    # Entity summary for prompt
    entity_summary = "; ".join(
        f"{k}: {', '.join(v)}"
        for k, v in entities.items()
        if v
    ) or "none detected"

    # Build medical prompt
    prompt = f"""You are a knowledgeable medical information assistant trained on NIH datasets.
You answer medical questions clearly and accurately using the provided MedQuAD knowledge.

IMPORTANT DISCLAIMER: Always remind the user to consult a qualified healthcare professional
for personal medical advice. You provide information, not medical diagnosis.

══ CONVERSATION HISTORY ══
{history_text or "(First message)"}

══ DETECTED MEDICAL ENTITIES ══
{entity_summary}

══ MEDQUAD KNOWLEDGE BASE ══
{medquad_context or "(No relevant MedQuAD documents found — answer from general knowledge.)"}

══ PATIENT QUESTION ══
{question}

══ INSTRUCTIONS ══
Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "answer": "Clear, empathetic answer grounded in MedQuAD. End with a brief disclaimer.",
  "confidence": 0.0,
  "reasoning_steps": [
    "Step 1 — which entity type was detected",
    "Step 2 — what MedQuAD evidence was found",
    "Step 3 — how the answer was composed"
  ],
  "sources_used": ["MedQuAD source labels used"],
  "context_reference": "Which part of the conversation this relates to, if any",
  "disclaimer": "Always consult a qualified healthcare professional for personal medical advice."
}}

Set confidence: 0.85+ when MedQuAD has a direct answer, 0.6-0.85 for partial, <0.6 for general knowledge.
"""

    # Call LLM
    parsed = {}
    try:
        resp = llm.invoke(prompt)
        raw = resp.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(raw[start:end])
        else:
            raise ValueError("No JSON found")
    except Exception as e:
        logger.error(f"❌ Medical LLM call failed: {e}")
        parsed = {
            "answer": (
                "I encountered an error retrieving medical information. "
                "Please consult a healthcare professional for reliable advice."
            ),
            "confidence": 0.3,
            "reasoning_steps": ["LLM call failed."],
            "sources_used": [],
            "context_reference": "",
            "disclaimer": "Always consult a qualified healthcare professional.",
        }

    # Validation pass (lightweight)
    answer     = parsed.get("answer", "")
    confidence = float(parsed.get("confidence", 0.6))

    disclaimer = parsed.get(
        "disclaimer",
        "⚕️ Always consult a qualified healthcare professional for personal medical advice.",
    )
    if disclaimer.lower() not in answer.lower():
        answer = answer.rstrip() + f"\n\n⚕️ *{disclaimer}*"

    return {
        "answer":            answer,
        "confidence":        confidence,
        "reasoning_steps":   parsed.get("reasoning_steps", []),
        "sources_used":      parsed.get("sources_used", []),
        "visual_tags":       [],
        "is_valid":          True,
        "validation_issues": [],
        "is_ambiguous":      False,
        "clarifications":    [],
        "context_reference": parsed.get("context_reference", ""),
        "medical_entities":  entities,
        "medquad_sources":   medquad_sources,
    }


if __name__ == "__main__":
    q = "What are the symptoms and treatments for type 2 diabetes?"
    ents = extract_medical_entities(q)
    print("NER result:", json.dumps(ents, indent=2))

    result = get_medical_response(q, conversation_history=[])
    print("\nMedical response:")
    print(json.dumps(result, indent=2, default=str))
