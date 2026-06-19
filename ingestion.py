import os
import io
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from PIL import Image

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv(r"c:\customer_service_chatbot_LLM\.env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths

VECTORDB_PATH    = r"c:\customer_service_chatbot_LLM\src\faiss_index"
CSV_PATH         = r"c:\customer_service_chatbot_LLM\src\dataset.csv"
SEEN_HASHES_PATH = r"c:\customer_service_chatbot_LLM\src\seen_hashes.json"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
genai.configure(api_key=GOOGLE_API_KEY)
_vision_model = genai.GenerativeModel("gemini-2.0-flash")


# Hash helpers (deduplication)

def load_seen_hashes() -> set:
    """Load previously ingested chunk hashes from disk."""
    if os.path.exists(SEEN_HASHES_PATH):
        with open(SEEN_HASHES_PATH, "r") as f:
            return set(json.load(f))
    return set()


def save_seen_hashes(hashes: set):
    """Persist the set of seen hashes to disk."""
    with open(SEEN_HASHES_PATH, "w") as f:
        json.dump(list(hashes), f)


def hash_doc(doc) -> str:
    return hashlib.md5(doc.page_content.strip().encode()).hexdigest()


# Text loaders

def load_from_csv(csv_path: str = CSV_PATH):
    """Load Q&A pairs from a CSV file."""
    try:
        loader = CSVLoader(file_path=csv_path, source_column="prompt")
        docs = loader.load()
        logger.info(f"📄 CSV: loaded {len(docs)} documents from {csv_path}")
        return docs
    except Exception as e:
        logger.error(f"❌ CSV load failed: {e}")
        return []


def load_from_urls(urls: list[str]):
    """Load documents from web URLs."""
    all_docs = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            all_docs.extend(docs)
            logger.info(f"🌐 Web: loaded {len(docs)} docs from {url}")
        except Exception as e:
            logger.error(f"❌ Web load failed for {url}: {e}")
    return all_docs


def split_documents(docs, chunk_size=500, chunk_overlap=50):
    """Split long documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"✂️  Split into {len(chunks)} chunks")
    return chunks


# Image ingestion

def extract_text_from_image(image_bytes: bytes, source_name: str = "image") -> Optional[Document]:
    """
    Use Gemini Vision to extract a rich textual description from an image,
    then wrap it as a LangChain Document so it can be embedded and stored
    in the FAISS index like any other text chunk.

    Args:
        image_bytes:  Raw bytes of the image file
        source_name:  Label stored in document metadata (e.g. filename)

    Returns:
        A Document, or None on failure.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        prompt = """Extract all information from this image for a customer service knowledge base.
Include:
1. All visible text (verbatim)
2. A detailed description of the image content
3. Any product names, codes, prices, or key data points
4. Context that would help answer customer questions

Write a single, coherent paragraph that combines all of the above."""

        response = _vision_model.generate_content([prompt, image])
        extracted = response.text.strip()
        if not extracted:
            return None

        doc = Document(
            page_content=extracted,
            metadata={"source": source_name, "type": "image_extraction"},
        )
        logger.info(f"🖼️  Image ingested: {source_name} → {len(extracted)} chars extracted")
        return doc
    except Exception as e:
        logger.error(f"❌ Image extraction failed for {source_name}: {e}")
        return None


def ingest_images(image_paths: list[str]) -> list[Document]:
    """Ingest a list of image file paths into Document objects."""
    docs = []
    for path in image_paths:
        try:
            with open(path, "rb") as f:
                image_bytes = f.read()
            doc = extract_text_from_image(
                image_bytes,
                source_name=Path(path).name,
            )
            if doc:
                docs.append(doc)
        except Exception as e:
            logger.error(f"❌ Failed to read image file {path}: {e}")
    logger.info(f"🖼️  Image ingestion complete — {len(docs)}/{len(image_paths)} succeeded")
    return docs


def ingest_image_bytes(image_bytes: bytes, filename: str = "uploaded_image") -> list[Document]:
    """Ingest a single image provided as raw bytes (e.g. from Streamlit uploader)."""
    doc = extract_text_from_image(image_bytes, source_name=filename)
    return [doc] if doc else []


# Core update function

def update_knowledge_base(
    extra_csv_paths: list[str] = None,
    web_urls: list[str] = None,
    image_paths: list[str] = None,
    image_bytes_list: list[tuple] = None,
) -> int:
    """
    Main ingestion pipeline:
      1. Load from CSV(s) + web URLs
      2. Load from image files/bytes
      3. Split long documents into chunks
      4. Deduplicate against seen hashes
      5. Upsert new chunks into FAISS
      6. Persist updated index + hashes

    Args:
        extra_csv_paths:    Additional CSV file paths beyond the default dataset.csv
        web_urls:           List of URLs to scrape and ingest
        image_paths:        List of local image file paths to ingest
        image_bytes_list:   List of (bytes, filename) tuples for in-memory images

    Returns:
        Number of new chunks added.
    """
    logger.info("🔄 Knowledge base update started…")

    # 1. Load text sources
    all_docs = load_from_csv(CSV_PATH)

    if extra_csv_paths:
        for path in extra_csv_paths:
            all_docs.extend(load_from_csv(path))

    if web_urls:
        web_docs = load_from_urls(web_urls)
        all_docs.extend(split_documents(web_docs))

    # 2. Load image sources
    if image_paths:
        img_docs = ingest_images(image_paths)
        all_docs.extend(img_docs)

    if image_bytes_list:
        for img_bytes, fname in image_bytes_list:
            docs = ingest_image_bytes(img_bytes, filename=fname)
            all_docs.extend(docs)

    if not all_docs:
        logger.warning("⚠️ No documents loaded. Skipping update.")
        return 0

    # 3. Deduplicate
    seen_hashes = load_seen_hashes()
    new_docs = []
    for doc in all_docs:
        h = hash_doc(doc)
        if h not in seen_hashes:
            new_docs.append(doc)
            seen_hashes.add(h)

    if not new_docs:
        logger.info("ℹ️ No new content found. Vector DB is already up to date.")
        return 0

    logger.info(
        f"✅ {len(new_docs)} new chunks to add "
        f"(skipped {len(all_docs) - len(new_docs)} duplicates)"
    )

    # 4. Upsert into FAISS
    if os.path.exists(VECTORDB_PATH):
        vectordb = FAISS.load_local(
            VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True
        )
        vectordb.add_documents(new_docs)
        logger.info("📥 Merged new docs into existing FAISS index")
    else:
        vectordb = FAISS.from_documents(new_docs, embeddings)
        logger.info("🆕 Created new FAISS index")

    # 5. Persist
    vectordb.save_local(VECTORDB_PATH)
    save_seen_hashes(seen_hashes)
    logger.info(f"💾 Saved FAISS index to {VECTORDB_PATH}")

    return len(new_docs)


if __name__ == "__main__":
    added = update_knowledge_base(
        web_urls=["https://example.com/faq"],
    )
    print(f"\n✅ Done — {added} new chunks added to knowledge base.")
