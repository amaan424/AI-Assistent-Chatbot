import os
import re
import time
import json
import base64
import logging
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from PIL import Image
import io

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Multilingual: open-source language detection + translation

try:
    from langdetect import detect_langs, LangDetectException
    _LANGDETECT_OK = True
except ImportError:
    _LANGDETECT_OK = False
    logger.warning("langdetect not installed — multilingual detection disabled")

try:
    from deep_translator import GoogleTranslator
    _TRANSLATOR_OK = True
except ImportError:
    _TRANSLATOR_OK = False
    logger.warning("deep-translator not installed — translation disabled")

SUPPORTED_LANGUAGES = {
    "en":    {"name": "English",              "native": "English",   "flag": "🇬🇧"},
    "hi":    {"name": "Hindi",                "native": "हिन्दी",     "flag": "🇮🇳"},
    "ar":    {"name": "Arabic",               "native": "العربية",   "flag": "🇸🇦"},
    "fr":    {"name": "French",               "native": "Français",  "flag": "🇫🇷"},
    "es":    {"name": "Spanish",              "native": "Español",   "flag": "🇪🇸"},
    "de":    {"name": "German",               "native": "Deutsch",   "flag": "🇩🇪"},
    "zh-cn": {"name": "Chinese (Simplified)", "native": "中文",       "flag": "🇨🇳"},
    "ja":    {"name": "Japanese",             "native": "日本語",     "flag": "🇯🇵"},
    "pt":    {"name": "Portuguese",           "native": "Português", "flag": "🇧🇷"},
    "ru":    {"name": "Russian",              "native": "Русский",   "flag": "🇷🇺"},
}

_DEEP_TRANSLATOR_MAP = {
    "en": "english", "hi": "hindi", "ar": "arabic", "fr": "french",
    "es": "spanish", "de": "german", "zh-cn": "chinese (simplified)",
    "ja": "japanese", "pt": "portuguese", "ru": "russian",
}

_LANGDETECT_REMAP = {"zh-tw": "zh-cn", "zh": "zh-cn"}


def detect_language(text: str) -> dict:
    """
    Detect language using langdetect (offline, open-source).

    Returns {"lang": str, "confidence": float, "flag": str, "name": str}
    Falls back to English on any failure.
    """
    if not _LANGDETECT_OK or not text or len(text.strip()) < 3:
        return {"lang": "en", "confidence": 1.0, "flag": "🇬🇧", "name": "English"}
    try:
        probs = detect_langs(text)
        top = probs[0]
        lang = _LANGDETECT_REMAP.get(top.lang, top.lang)
        if float(top.prob) < 0.55 and len(text.split()) <= 4:
            lang = "en"
        if lang not in SUPPORTED_LANGUAGES:
            lang = "en"
        info = SUPPORTED_LANGUAGES[lang]
        return {
            "lang": lang,
            "confidence": float(top.prob),
            "flag": info["flag"],
            "name": info["name"],
        }
    except Exception:
        return {"lang": "en", "confidence": 1.0, "flag": "🇬🇧", "name": "English"}


def translate_to_english(text: str, source_lang: str) -> str:
    """
    Translate text from source_lang to English using deep-translator.
    Returns original text on any failure or if already English.
    """
    if source_lang == "en" or not _TRANSLATOR_OK:
        return text
    src = _DEEP_TRANSLATOR_MAP.get(source_lang, source_lang)
    try:
        return GoogleTranslator(source=src, target="english").translate(text) or text
    except Exception as e:
        logger.warning(f"Translation to English failed ({source_lang}→en): {e}")
        return text


def translate_from_english(text: str, target_lang: str) -> str:
    """
    Translate text from English to target_lang using deep-translator.
    Returns original text on any failure or if target is English.
    """
    if target_lang == "en" or not _TRANSLATOR_OK:
        return text
    tgt = _DEEP_TRANSLATOR_MAP.get(target_lang, target_lang)
    try:
        if len(text) <= 4800:
            return GoogleTranslator(source="english", target=tgt).translate(text) or text
        words = text.split()
        chunks, cur, length = [], [], 0
        for w in words:
            if length + len(w) + 1 > 4800 and cur:
                chunks.append(" ".join(cur)); cur, length = [], 0
            cur.append(w); length += len(w) + 1
        if cur:
            chunks.append(" ".join(cur))
        translated_chunks = [
            GoogleTranslator(source="english", target=tgt).translate(c) or c
            for c in chunks
        ]
        return " ".join(translated_chunks)
    except Exception as e:
        logger.warning(f"Translation from English failed (en→{target_lang}): {e}")
        return text


def detect_mixed_language(text: str, primary_lang: str) -> bool:
    """
    Returns True if the text appears to contain segments in more than one language.
    Uses a simple heuristic: split on punctuation, detect each segment, compare.
    """
    if not _LANGDETECT_OK or len(text.strip()) < 15:
        return False
    try:
        segments = [s.strip() for s in re.split(r"[.!?।؟\n]+", text) if len(s.strip()) > 8]
        if len(segments) < 2:
            return False
        langs = set()
        for seg in segments:
            d = detect_language(seg)
            if d["confidence"] > 0.6:
                langs.add(d["lang"])
        return len(langs) > 1
    except Exception:
        return False


# Model setup

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
genai.configure(api_key=GOOGLE_API_KEY)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1,
)

vision_model = genai.GenerativeModel("gemini-2.5-flash-lite")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb_file_path = str(BASE_DIR / "faiss_index")


# Retry helper — absorbs transient Gemini rate-limit / quota / timeout errors

def _call_with_retry(fn, *args, max_attempts: int = 3, base_delay: float = 1.5, **kwargs):
    """
    Calls fn(*args, **kwargs) with exponential-backoff retry.

    Free-tier Gemini keys return transient 429 (rate limit) / 503 (overloaded)
    errors fairly easily. Without retry, a single transient hit makes the entire
    turn fall back to an error response. Retrying once or twice with a short
    delay clears most of these silently.
    """
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            err_text = str(e).lower()
            transient = any(
                marker in err_text
                for marker in ["429", "resourceexhausted", "503", "unavailable", "rate limit", "quota", "deadline"]
            )
            if attempt < max_attempts and transient:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"⏳ Gemini call hit a transient error (attempt {attempt}/{max_attempts}): {e} "
                    f"— retrying in {delay:.1f}s"
                )
                time.sleep(delay)
                continue
            break
    raise last_err


# 1. KB creation

def create_vector_db():
    """Initial creation of the vector DB from the base CSV dataset."""
    from ingestion import update_knowledge_base
    added = update_knowledge_base()
    return added


# 2. Image analysis — extract structured info from uploaded image

def analyze_image(image_bytes: bytes, user_question: str) -> dict:
    """
    Send image + question to Gemini Vision.

    Returns:
        {
            "description":     str — what the image contains,
            "extracted_text":  str — any text visible in the image,
            "visual_tags":     list — key objects / concepts detected,
            "answer":          str — direct answer to user_question from the image,
            "confidence":      float — 0-1 self-reported confidence
        }
    """
    logger.info("🖼️  Running vision analysis…")
    image = Image.open(io.BytesIO(image_bytes))

    vision_prompt = f"""You are a precise visual analyst. Examine the image carefully.

User question: "{user_question}"

Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "description": "One paragraph describing what the image shows",
  "extracted_text": "All text visible in the image, or empty string if none",
  "visual_tags": ["tag1", "tag2", "tag3"],
  "answer": "Direct answer to the user question based purely on what you see",
  "confidence": 0.0
}}

Set confidence between 0.0 (completely uncertain) and 1.0 (completely certain).
"""
    try:
        response = _call_with_retry(vision_model.generate_content, [vision_prompt, image])
        raw = response.text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
        result.setdefault("confidence", 0.6)
        result.setdefault("visual_tags", [])
        result.setdefault("extracted_text", "")
        logger.info(f"✅ Vision analysis done — tags: {result['visual_tags']}")
        return result
    except Exception as e:
        logger.error(f"❌ Vision analysis failed: {e}")
        return {
            "description": "Image could not be analysed.",
            "extracted_text": "",
            "visual_tags": [],
            "answer": "I was unable to process the image. Please try again.",
            "confidence": 0.0,
        }


# 3. Ambiguity detection — classify query before spending retrieval budget

def detect_ambiguity(question: str, has_image: bool = False) -> dict:
    """
    Calls the LLM to determine whether the question has multiple valid
    interpretations that require clarification.

    Returns:
        {
            "is_ambiguous": bool,
            "clarifications": list[str],
            "reason": str
        }
    """
    if len(question.strip()) < 8:
        return {"is_ambiguous": False, "clarifications": [], "reason": "Too short to be ambiguous."}

    prompt = f"""You are a query classifier. Determine whether this customer-service question is ambiguous.
A question is ambiguous when it has 2+ very different valid interpretations that would require
genuinely different knowledge-base lookups.

Do NOT mark a message as ambiguous just because:
- it is a complaint, vent, or expression of frustration without a specific question
  (e.g. "this is ridiculous, nothing works") — treat that as a request for help with
  whatever issue they're describing, not an ambiguity case.
- it is vague but only has ONE reasonable interpretation in a customer-service context.
Only flag is_ambiguous=true when you can articulate 2+ clearly distinct, specific interpretations.

Has image attached: {has_image}
Question: "{question}"

Reply ONLY with valid JSON:
{{
  "is_ambiguous": true or false,
  "clarifications": ["Rephrased version A (full question)", "Rephrased version B (full question)"],
  "reason": "One sentence explaining your decision"
}}

If not ambiguous, clarifications must be an empty list [].
"""
    try:
        resp = _call_with_retry(llm.invoke, prompt)
        raw = resp.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
        logger.info(f"🔍 Ambiguity check — is_ambiguous={result.get('is_ambiguous')}")
        return result
    except Exception as e:
        logger.warning(f"⚠️ Ambiguity detection failed ({e}); defaulting to unambiguous.")
        return {"is_ambiguous": False, "clarifications": [], "reason": "Detection failed."}


# 3B. Sentiment analysis — detect customer emotion before generating a reply

def analyze_sentiment(question: str, conversation_history: list) -> dict:
    """
    Calls Gemini to classify the emotional tone of the customer's current
    message. The last couple of turns are included as context only.

    Returns:
        {
            "sentiment":       "positive" | "negative" | "neutral",
            "sentiment_score": float in [-1.0, 1.0],
            "emotion":         str — one of a fixed set,
            "intensity":       "low" | "medium" | "high",
            "escalate":        bool — True if a human hand-off is advisable,
            "reason":          str — one-line justification
        }
    """
    history_snippet = ""
    for msg in conversation_history[-4:]:
        role = "Customer" if msg["role"] == "user" else "Assistant"
        history_snippet += f"{role}: {msg['content']}\n"

    prompt = f"""You are a sentiment-analysis engine for a customer service chatbot.
Classify the emotional tone of the CUSTOMER'S latest message only. Use the
recent conversation purely as context — do not classify the assistant's tone.

Recent conversation:
{history_snippet or "(No prior turns)"}

Customer's latest message: "{question}"

Reply ONLY with valid JSON (no markdown, no extra text):
{{
  "sentiment": "positive" or "negative" or "neutral",
  "sentiment_score": 0.0,
  "emotion": "happy" or "satisfied" or "neutral" or "confused" or "frustrated" or "disappointed" or "angry" or "anxious",
  "intensity": "low" or "medium" or "high",
  "escalate": true or false,
  "reason": "One short sentence explaining the classification"
}}

Guidance:
- sentiment_score ranges from -1.0 (extremely negative) to 1.0 (extremely positive); 0.0 is neutral.
- Set escalate=true ONLY when the customer is angry/frustrated at high intensity, mentions repeated
  failed attempts, threatens to cancel/leave/sue, or explicitly asks for a human or manager.
- A confused or mildly negative message is NOT automatically an escalation.
"""
    try:
        resp = _call_with_retry(llm.invoke, prompt)
        raw = resp.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
        result.setdefault("sentiment", "neutral")
        result.setdefault("sentiment_score", 0.0)
        result.setdefault("emotion", "neutral")
        result.setdefault("intensity", "low")
        result.setdefault("escalate", False)
        result["sentiment_score"] = float(result["sentiment_score"])
        logger.info(
            f"🎭 Sentiment — {result['sentiment']} ({result['emotion']}, "
            f"score={result['sentiment_score']:.2f}, escalate={result['escalate']})"
        )
        return result
    except Exception as e:
        logger.warning(f"⚠️ Sentiment analysis failed ({e}); defaulting to neutral.")
        return {
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "emotion": "neutral",
            "intensity": "low",
            "escalate": False,
            "reason": "Detection failed — defaulted to neutral.",
        }


# 3C. Combined classifier — ambiguity + sentiment in a single Gemini call

def classify_message(question: str, conversation_history: list, has_image: bool = False, skip_ambiguity: bool = False) -> dict:
    """
    Runs ambiguity detection AND sentiment analysis in a single Gemini call.

    Merging two classification calls into one keeps each chat turn at 3 total
    Gemini calls (classify → generate → validate, plus vision if an image is
    attached) instead of 4. This matters on rate-limited free-tier API keys.

    Returns a dict combining both result shapes:
        {
            "is_ambiguous": bool, "clarifications": list[str], "ambiguity_reason": str,
            "sentiment": "positive"|"negative"|"neutral", "sentiment_score": float,
            "emotion": str, "intensity": "low"|"medium"|"high", "escalate": bool,
            "sentiment_reason": str,
        }
    """
    if len(question.strip()) < 3:
        return {
            "is_ambiguous": False, "clarifications": [], "ambiguity_reason": "Too short to classify.",
            "sentiment": "neutral", "sentiment_score": 0.0, "emotion": "neutral",
            "intensity": "low", "escalate": False, "sentiment_reason": "Too short to classify.",
        }

    # Fast path: greetings / small talk are never ambiguous, and don't need an
    # LLM call to classify — saves quota and avoids the model inventing two
    # "interpretations" of what is just a hello.
    GREETING_PATTERN = re.compile(
        r"^\s*(hi+|hey+|hello+|yo+|sup|greetings|good\s*(morning|afternoon|evening|day))"
        r"\s*(there|chatbot|bot|assistant|team|everyone)?\s*[!.?]*\s*$",
        re.IGNORECASE,
    )
    if GREETING_PATTERN.match(question.strip()):
        return {
            "is_ambiguous": False, "clarifications": [], "ambiguity_reason": "Greeting — no clarification needed.",
            "sentiment": "positive", "sentiment_score": 0.3, "emotion": "happy",
            "intensity": "low", "escalate": False, "sentiment_reason": "Friendly greeting.",
        }

    history_snippet = ""
    for msg in conversation_history[-4:]:
        role = "Customer" if msg["role"] == "user" else "Assistant"
        history_snippet += f"{role}: {msg['content']}\n"

    prompt = f"""You are a combined query-classifier and sentiment-analysis engine for a
customer service chatbot. Given the customer's latest message, do BOTH of the following
in a single pass:

1) AMBIGUITY — Decide whether the message has 2+ very different valid interpretations that
   would require genuinely different knowledge-base lookups.
   Do NOT mark it ambiguous just because it's a complaint/vent without a specific question
   (e.g. "this is ridiculous, nothing works" — that's a request for help, not ambiguity), or
   because it's vague but only has ONE reasonable reading in context. Only flag
   is_ambiguous=true when the 2+ interpretations would lead to DIFFERENT knowledge-base
   lookups or DIFFERENT answers. If your two "clarifications" are just two ways of phrasing
   the same underlying question, they are NOT distinct — mark is_ambiguous=false.
   BAD example (do not do this): question "Are you just saying hello?" →
   clarifications ["Are you simply greeting me?", "Are you asking if I am saying hello?"]
   — these are the same interpretation reworded twice, not two distinct ones. Correct
   answer for that question is is_ambiguous=false.
   Greetings and small talk directed at the assistant (e.g. "hello chatbot", "hi there",
   "good morning") are NEVER ambiguous, no matter how you phrase potential interpretations
   of them — always mark is_ambiguous=false for these.

2) SENTIMENT — Classify the emotional tone of the CUSTOMER'S latest message. Use the recent
   conversation only as context; do not classify the assistant's tone.

Has image attached: {has_image}
Recent conversation:
{history_snippet or "(No prior turns)"}
Customer's latest message: "{question}"

Reply ONLY with valid JSON (no markdown, no extra text):
{{
  "is_ambiguous": true or false,
  "clarifications": ["Rephrased version A (full question)", "Rephrased version B (full question)"],
  "ambiguity_reason": "One sentence explaining the ambiguity decision",
  "sentiment": "positive" or "negative" or "neutral",
  "sentiment_score": 0.0,
  "emotion": "happy" or "satisfied" or "neutral" or "confused" or "frustrated" or "disappointed" or "angry" or "anxious",
  "intensity": "low" or "medium" or "high",
  "escalate": true or false,
  "sentiment_reason": "One sentence explaining the sentiment decision"
}}

Guidance:
- If not ambiguous, clarifications must be an empty list [].
- sentiment_score ranges from -1.0 (extremely negative) to 1.0 (extremely positive); 0.0 is neutral.
- Set escalate=true ONLY when the customer is angry/frustrated at high intensity, mentions repeated
  failed attempts, threatens to cancel/leave/sue, or explicitly asks for a human or manager.
- A confused or mildly negative message is NOT automatically an escalation.
"""
    try:
        resp = _call_with_retry(llm.invoke, prompt)
        raw = resp.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        result = json.loads(raw[start:end] if start != -1 and end > start else raw)

        result.setdefault("is_ambiguous", False)
        result.setdefault("clarifications", [])
        result.setdefault("ambiguity_reason", "")
        result.setdefault("sentiment", "neutral")
        result.setdefault("sentiment_score", 0.0)
        result.setdefault("emotion", "neutral")
        result.setdefault("intensity", "low")
        result.setdefault("escalate", False)
        result["sentiment_score"] = float(result["sentiment_score"])

        # Hard override: if this message is itself the user's reply to a clarification
        # we just asked (e.g. they clicked one of our own suggested clarification
        # buttons), never flag it ambiguous again — otherwise the bot can loop forever
        # asking the user to clarify their own clarification.
        if skip_ambiguity:
            if result["is_ambiguous"]:
                logger.info("🛡️ Skipping ambiguity flag — this is a reply to a prior clarification prompt")
            result["is_ambiguous"] = False
            result["clarifications"] = []
            result["ambiguity_reason"] = "Skipped: reply to a previous clarification prompt."

        # Safety net: override false-positive ambiguity flags where the model's
        # two "clarifications" are just paraphrases of each other rather than
        # genuinely distinct interpretations (common failure mode on small models).
        if result["is_ambiguous"] and len(result["clarifications"]) >= 2:
            words_a = set(re.findall(r"\w+", result["clarifications"][0].lower()))
            words_b = set(re.findall(r"\w+", result["clarifications"][1].lower()))
            if words_a and words_b:
                overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
                if overlap > 0.6:
                    logger.info(f"🛡️ Overriding false-positive ambiguity (word overlap={overlap:.2f})")
                    result["is_ambiguous"] = False
                    result["clarifications"] = []
                    result["ambiguity_reason"] = "Overridden: clarifications were near-duplicates."

        logger.info(
            f"🔍 Classify — ambiguous={result['is_ambiguous']}, "
            f"sentiment={result['sentiment']} ({result['emotion']}), escalate={result['escalate']}"
        )
        return result
    except Exception as e:
        logger.warning(f"⚠️ Combined classification failed ({e}); defaulting to safe values.")
        return {
            "is_ambiguous": False, "clarifications": [], "ambiguity_reason": "Detection failed.",
            "sentiment": "neutral", "sentiment_score": 0.0, "emotion": "neutral",
            "intensity": "low", "escalate": False, "sentiment_reason": "Detection failed.",
        }


# 4. Contextual reasoning — build rich context from history + retrieved docs

def build_contextual_prompt(
    question: str,
    retrieved_docs: list,
    conversation_history: list,
    image_analysis: Optional[dict] = None,
    sentiment: Optional[dict] = None,
    user_lang: str = "en",
    english_question: str = "",
) -> str:
    """
    Composes the final generation prompt with:
      - Multi-turn conversation history (last 6 turns)
      - Retrieved KB chunks with source labels
      - Optional visual context from image analysis
      - Optional sentiment/tone guidance
      - Instruction to produce a structured JSON response
    """
    history_text = ""
    for msg in conversation_history[-6:]:
        role = "Customer" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    kb_context = ""
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source", f"doc_{i}")
        kb_context += f"[Source {i+1} — {source}]\n{doc.page_content.strip()}\n\n"

    visual_block = ""
    if image_analysis:
        visual_block = f"""
── VISUAL CONTEXT (from uploaded image) ──
Description  : {image_analysis.get('description', '')}
Extracted text: {image_analysis.get('extracted_text', '')}
Visual tags  : {', '.join(image_analysis.get('visual_tags', []))}
Vision answer: {image_analysis.get('answer', '')}
──────────────────────────────────────────
"""

    tone_block = ""
    if sentiment:
        tone_by_sentiment = {
            "positive": "The customer sounds happy/satisfied. Match their warmth — be "
                        "friendly and conversational, and reinforce the positive experience.",
            "negative": "The customer sounds unhappy. Open with a brief, genuine acknowledgement "
                        "of their frustration before getting to the resolution. Be concise and "
                        "sincere — do not sound scripted, and do not over-apologise.",
            "neutral":  "The customer's tone is neutral. Stay clear, professional, and efficient.",
        }
        tone_block = f"""
── CUSTOMER SENTIMENT ──
Detected sentiment : {sentiment.get('sentiment', 'neutral')} (emotion: {sentiment.get('emotion', 'neutral')}, intensity: {sentiment.get('intensity', 'low')})
Tone guidance       : {tone_by_sentiment.get(sentiment.get('sentiment', 'neutral'), tone_by_sentiment['neutral'])}"""
        if sentiment.get("escalate"):
            tone_block += (
                "\n⚠️ This customer may need human escalation — clearly acknowledge their "
                "frustration and mention you can connect them with a human agent if this "
                "doesn't fully resolve things."
            )
        tone_block += "\n──────────────────────────────────────────\n"

    lang_block = ""
    if user_lang != "en":
        lang_info = SUPPORTED_LANGUAGES.get(user_lang, {})
        lang_block = f"""
── LANGUAGE CONTEXT ──
User is writing in: {lang_info.get('flag', '')} {lang_info.get('name', user_lang)} (ISO: {user_lang})
English interpretation of their question: "{english_question or question}"
CRITICAL: You MUST write your entire answer in {lang_info.get('name', user_lang)}.
Ground all claims in the KB context above. If you must include a technical term
with no good translation, write it in English inside parentheses.
──────────────────────────────────────────
"""

    prompt = f"""You are a knowledgeable, evidence-based customer service assistant.
You MUST ground every claim in the provided knowledge base or visual evidence.
If the answer is not in the context, say "I don't know" — never fabricate.

══ CONVERSATION HISTORY (recent turns) ══
{history_text or "(First message — no history yet)"}

══ KNOWLEDGE BASE CONTEXT ══
{kb_context or "(No relevant documents retrieved)"}
{visual_block}
{tone_block}{lang_block}
══ CURRENT QUESTION ══
{question}

══ INSTRUCTIONS ══
Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "answer": "Your full, friendly answer{' — written entirely in ' + SUPPORTED_LANGUAGES.get(user_lang, {}).get('name', user_lang) if user_lang != 'en' else ''}",
  "confidence": 0.0,
  "reasoning_steps": [
    "Step 1 — what you looked at first",
    "Step 2 — what evidence you found",
    "Step 3 — how you formed your answer"
  ],
  "sources_used": ["Source 1 label", "Source 2 label"],
  "context_reference": "Brief note on which prior turn (if any) this relates to"
}}

Set confidence 0.9+ for factual certainty, 0.7–0.9 for well-supported, 0.4–0.7 for partial, <0.4 for speculative.
"""
    return prompt


# 5. Response validation — self-critique pass before returning to user

def validate_response(answer: str, question: str, kb_context: str) -> dict:
    """
    Runs a lightweight second-pass validation:
      - Does the answer actually address the question?
      - Is it grounded in the provided context?
      - Any obvious factual inconsistencies?

    Returns:
        {
            "is_valid": bool,
            "issues": list[str],
            "corrected_answer": str or None
        }
    """
    prompt = f"""You are a strict QA reviewer. Evaluate the answer below.

Question: "{question}"
Available context (summary): "{kb_context[:600]}"
Answer to review: "{answer[:800]}"

Reply ONLY with valid JSON:
{{
  "is_valid": true or false,
  "issues": ["Issue 1 if any", "Issue 2 if any"],
  "corrected_answer": null
}}

Mark is_valid=false only if the answer is clearly wrong, fabricated, or completely off-topic.
If you suggest a correction, put it in corrected_answer; otherwise null.
"""
    try:
        resp = _call_with_retry(llm.invoke, prompt)
        raw = resp.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
        logger.info(f"✅ Validation — is_valid={result.get('is_valid')}, issues={result.get('issues')}")
        return result
    except Exception as e:
        logger.warning(f"⚠️ Validation failed ({e}); marking as valid by default.")
        return {"is_valid": True, "issues": [], "corrected_answer": None}


# 6. Main multimodal QA pipeline

def get_multimodal_response(
    question: str,
    conversation_history: list,
    image_bytes: Optional[bytes] = None,
    skip_ambiguity: bool = False,
) -> dict:
    """
    Full pipeline:
      0. Language detection — identify user language (langdetect, open-source)
      1. Combined classify — ambiguity + sentiment in one Gemini call
      2. Translation — translate question to English for KB retrieval
      3. Image analysis — extract visual context (if image provided)
      4. KB retrieval — fetch relevant documents from FAISS (English query)
      5. Contextual prompt — build rich prompt with history + sentiment + vision + KB + language
      6. LLM generation — produce structured JSON response in user's language
      7. Validation pass — self-critique and optionally correct
      8. Return rich result dict

    Args:
        question:             User's text question (any supported language)
        conversation_history: List of {"role": "user"/"assistant", "content": str}
        image_bytes:          Raw image bytes if an image was uploaded

    Returns a dict with keys:
        answer, confidence, reasoning_steps, sources_used,
        visual_tags, is_valid, validation_issues,
        is_ambiguous, clarifications, context_reference,
        sentiment, sentiment_score, emotion, escalate,
        detected_lang, detected_lang_name, detected_lang_flag,
        lang_confidence, is_mixed_language
    """
    has_image = image_bytes is not None

    # Step 0: Language detection (open-source, no API call)
    logger.info("🔄 Step 0/8 — Language detection")
    lang_det = detect_language(question)
    user_lang = lang_det["lang"]
    is_mixed  = detect_mixed_language(question, user_lang)
    logger.info(f"🌐 Language: {lang_det['name']} ({user_lang}), conf={lang_det['confidence']:.2f}, mixed={is_mixed}")

    english_question = translate_to_english(question, user_lang)
    classification_input = english_question if user_lang != "en" else question

    # Step 1: Combined ambiguity + sentiment classification
    logger.info("🔄 Step 1/8 — Combined ambiguity + sentiment classification")
    classification = classify_message(classification_input, conversation_history, has_image=has_image, skip_ambiguity=skip_ambiguity)
    sentiment = {
        "sentiment": classification.get("sentiment", "neutral"),
        "sentiment_score": classification.get("sentiment_score", 0.0),
        "emotion": classification.get("emotion", "neutral"),
        "intensity": classification.get("intensity", "low"),
        "escalate": classification.get("escalate", False),
    }

    if classification.get("is_ambiguous"):
        if sentiment.get("sentiment") == "negative":
            clarification_intro_en = (
                "I can tell this has been frustrating, and I want to get you the right "
                "answer rather than guess. Could you help me narrow it down?"
            )
        else:
            clarification_intro_en = "Your question could be interpreted in more than one way. Could you clarify?"
        clarification_intro = translate_from_english(clarification_intro_en, user_lang)
        clarifications_en = classification.get("clarifications", [])
        clarifications = [translate_from_english(c, user_lang) for c in clarifications_en]
        return {
            "answer": clarification_intro,
            "confidence": 0.0,
            "reasoning_steps": [f"Ambiguity detected: {classification.get('ambiguity_reason')}"],
            "sources_used": [],
            "visual_tags": [],
            "is_valid": True,
            "validation_issues": [],
            "is_ambiguous": True,
            "clarifications": clarifications,
            "context_reference": "",
            "sentiment": sentiment.get("sentiment", "neutral"),
            "sentiment_score": sentiment.get("sentiment_score", 0.0),
            "emotion": sentiment.get("emotion", "neutral"),
            "escalate": sentiment.get("escalate", False),
            "detected_lang": user_lang,
            "detected_lang_name": lang_det["name"],
            "detected_lang_flag": lang_det["flag"],
            "lang_confidence": lang_det["confidence"],
            "is_mixed_language": is_mixed,
        }

    # Step 2: Image analysis
    image_analysis = None
    visual_tags = []
    if has_image:
        logger.info("🔄 Step 2/8 — Vision analysis")
        image_analysis = analyze_image(image_bytes, english_question)
        visual_tags = image_analysis.get("visual_tags", [])

    # Step 3: KB retrieval (always in English)
    logger.info("🔄 Step 3/8 — Knowledge base retrieval")
    retrieved_docs = []
    kb_context_raw = ""
    try:
        vectordb = FAISS.load_local(
            vectordb_file_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4, "score_threshold": 0.5},
        )
        retrieved_docs = retriever.invoke(english_question)
        kb_context_raw = " ".join(d.page_content for d in retrieved_docs)
        logger.info(f"📚 Retrieved {len(retrieved_docs)} docs from FAISS")
    except Exception as e:
        logger.warning(f"⚠️ KB retrieval failed: {e} — continuing without KB context")

    # Step 4: Build contextual prompt
    logger.info("🔄 Step 4/8 — Building contextual prompt")
    full_prompt = build_contextual_prompt(
        question=question,
        retrieved_docs=retrieved_docs,
        conversation_history=conversation_history,
        image_analysis=image_analysis,
        sentiment=sentiment,
        user_lang=user_lang,
        english_question=english_question,
    )

    # Step 5: LLM generation
    logger.info("🔄 Step 5/8 — LLM generation")
    raw_response = ""
    parsed = {}
    try:
        resp = _call_with_retry(llm.invoke, full_prompt)
        raw_response = resp.content.strip()
        raw_response = re.sub(r"```json|```", "", raw_response).strip()
        start = raw_response.find("{")
        end = raw_response.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(raw_response[start:end])
        else:
            raise ValueError("No JSON object found in response")
    except Exception as e:
        logger.error(f"❌ LLM generation/parsing failed: {e}")
        parsed = {
            "answer": raw_response or "I encountered an error generating a response.",
            "confidence": 0.3,
            "reasoning_steps": ["Generation failed — returning raw output."],
            "sources_used": [],
            "context_reference": "",
        }

    answer = parsed.get("answer", "")
    confidence = float(parsed.get("confidence", 0.6))
    reasoning_steps = parsed.get("reasoning_steps", [])
    sources_used = parsed.get("sources_used", [])
    context_reference = parsed.get("context_reference", "")

    # Step 6: Validation pass
    logger.info("🔄 Step 6/8 — Response validation")
    validation = validate_response(answer, question, kb_context_raw)
    if not validation.get("is_valid") and validation.get("corrected_answer"):
        answer = validation["corrected_answer"]
        confidence = max(0.0, confidence - 0.15)

    logger.info(
        f"🏁 Pipeline complete — lang={user_lang}, confidence={confidence:.2f}, "
        f"valid={validation.get('is_valid')}, sentiment={sentiment.get('sentiment')}"
    )

    return {
        "answer": answer,
        "confidence": confidence,
        "reasoning_steps": reasoning_steps,
        "sources_used": sources_used,
        "visual_tags": visual_tags,
        "is_valid": validation.get("is_valid", True),
        "validation_issues": validation.get("issues", []),
        "is_ambiguous": False,
        "clarifications": [],
        "context_reference": context_reference,
        "sentiment": sentiment.get("sentiment", "neutral"),
        "sentiment_score": sentiment.get("sentiment_score", 0.0),
        "emotion": sentiment.get("emotion", "neutral"),
        "escalate": sentiment.get("escalate", False),
        "detected_lang":      user_lang,
        "detected_lang_name": lang_det["name"],
        "detected_lang_flag": lang_det["flag"],
        "lang_confidence":    lang_det["confidence"],
        "is_mixed_language":  is_mixed,
        "english_question":   english_question if user_lang != "en" else "",
    }


# Legacy compatibility — keeps original get_qa_chain() working

def get_qa_chain():
    """
    Original single-turn RAG chain preserved for backward compatibility.
    New code should call get_multimodal_response() instead.
    """
    vectordb = FAISS.load_local(
        vectordb_file_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


if __name__ == "__main__":
    create_vector_db()
    result = get_multimodal_response(
        question="What is your return policy?",
        conversation_history=[],
    )
    print(json.dumps(result, indent=2))
