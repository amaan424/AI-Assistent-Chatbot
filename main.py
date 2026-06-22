import streamlit as st
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

from langchain_helper import get_multimodal_response, create_vector_db
from ingestion import update_knowledge_base
from medical_helper import (
    get_medical_response,
    extract_medical_entities,
    format_entities_for_display,
    build_medical_index,
    MEDICAL_FAISS_PATH,
    MEDICAL_META_PATH,
)
from arxiv_ui import render_arxiv_tab, init_arxiv_session_state
from arxiv_helper import build_arxiv_index, ARXIV_FAISS_PATH, ARXIV_META_PATH

import os
import json

# Page config

st.set_page_config(
    page_title="AI Assistant — Customer Service + Medical + arXiv Expert",
    page_icon="🤖",
    layout="wide",
)

# Session state initialisation

_defaults = {
    "scheduler":          None,
    "scheduler_running":  False,
    "last_update":        "Never",
    "update_log":         [],
    "chat_history":           [],
    "pending_clarifications": [],
    "pending_question":       "",
    "total_queries":          0,
    "image_queries":          0,
    "avg_confidence":         0.0,
    "confidence_sum":         0.0,
    "sentiment_counts":   {"positive": 0, "neutral": 0, "negative": 0},
    "sentiment_history":  [],
    "escalation_flag":    False,
    "med_chat_history":    [],
    "med_total_queries":   0,
    "med_avg_confidence":  0.0,
    "med_confidence_sum":  0.0,
    "med_last_entities":   {},
    "med_last_sources":    [],
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

init_arxiv_session_state()


# Scheduler helpers

def run_update(web_urls=None, extra_csvs=None):
    added = update_knowledge_base(extra_csv_paths=extra_csvs, web_urls=web_urls)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.last_update = timestamp
    st.session_state.update_log.append(f"[{timestamp}] ✅ Added {added} new chunks")


def start_scheduler(interval_hours: int, web_urls: list, extra_csvs: list):
    if st.session_state.scheduler_running:
        st.session_state.scheduler.shutdown(wait=False)
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        run_update,
        trigger="interval",
        hours=interval_hours,
        kwargs={"web_urls": web_urls, "extra_csvs": extra_csvs},
        id="kb_update_job",
        next_run_time=datetime.now(),
    )
    scheduler.start()
    st.session_state.scheduler = scheduler
    st.session_state.scheduler_running = True


def stop_scheduler():
    if st.session_state.scheduler:
        st.session_state.scheduler.shutdown(wait=False)
    st.session_state.scheduler_running = False


# UI helpers

def confidence_badge(score: float) -> str:
    if score >= 0.75:
        return f"🟢 High confidence ({score:.0%})"
    elif score >= 0.45:
        return f"🟡 Medium confidence ({score:.0%})"
    else:
        return f"🔴 Low confidence ({score:.0%})"


def sentiment_badge(sentiment: str, emotion: str = "") -> str:
    icons = {"positive": "😊", "neutral": "😐", "negative": "😟"}
    icon = icons.get(sentiment, "😐")
    label = sentiment.capitalize() if sentiment else "Neutral"
    if emotion and emotion.lower() != sentiment:
        return f"{icon} {label} ({emotion})"
    return f"{icon} {label}"


def render_assistant_message(msg: dict):
    """Render a rich assistant message bubble for the chatbot tab."""
    with st.chat_message("assistant"):
        if msg.get("escalate"):
            st.warning(
                "🚨 This customer seems highly frustrated — consider connecting them with a human agent."
            )
        if msg.get("is_mixed_language"):
            st.info("🔀 Mixed-language input detected — responding in the primary detected language.")
        st.write(f"🤖 {msg['content']}")
        meta_cols = st.columns(5)
        with meta_cols[0]:
            st.caption(confidence_badge(msg.get("confidence", 0.6)))
        with meta_cols[1]:
            st.caption(sentiment_badge(msg.get("sentiment", "neutral"), msg.get("emotion", "")))
        with meta_cols[2]:
            if msg.get("is_valid", True):
                st.caption("✅ Response validated")
            else:
                st.caption("⚠️ Validation issues found")
        with meta_cols[3]:
            if msg.get("visual_tags"):
                st.caption(f"👁️ Visual: {', '.join(msg['visual_tags'][:3])}")
        with meta_cols[4]:
            if msg.get("context_reference"):
                st.caption(f"🔗 {msg['context_reference']}")
        flag = msg.get("detected_lang_flag", "")
        lang_name = msg.get("detected_lang_name", "")
        lang_conf = msg.get("lang_confidence", 0.0)
        if flag and lang_name:
            eng_q = msg.get("english_question", "")
            label = f"{flag} {lang_name} ({lang_conf:.0%})"
            if eng_q:
                label += f" · translated: \"{eng_q[:60]}{'…' if len(eng_q) > 60 else ''}\""
            st.caption(f"🌐 {label}")
        if msg.get("sources_used"):
            st.caption("📎 **Sources:** " + " · ".join(msg["sources_used"]))
        if msg.get("validation_issues"):
            with st.expander("⚠️ Validation notes", expanded=False):
                for issue in msg["validation_issues"]:
                    st.write(f"• {issue}")
        if msg.get("reasoning_steps"):
            with st.expander("🧠 Reasoning trace", expanded=False):
                for i, step in enumerate(msg["reasoning_steps"], 1):
                    st.write(f"**Step {i}:** {step}")


def render_medical_message(msg: dict):
    """Render a rich medical assistant message bubble."""
    with st.chat_message("assistant"):
        st.markdown(f"⚕️ {msg['content']}")
        meta_cols = st.columns(3)
        with meta_cols[0]:
            st.caption(confidence_badge(msg.get("confidence", 0.6)))
        with meta_cols[1]:
            sources = msg.get("sources_used", [])
            if sources:
                st.caption(f"📚 {len(sources)} MedQuAD source(s)")
        with meta_cols[2]:
            if msg.get("context_reference"):
                st.caption(f"🔗 {msg['context_reference']}")
        if msg.get("reasoning_steps"):
            with st.expander("🧠 Reasoning trace", expanded=False):
                for i, step in enumerate(msg["reasoning_steps"], 1):
                    st.write(f"**Step {i}:** {step}")
        if msg.get("medquad_sources"):
            with st.expander("📖 MedQuAD citations", expanded=False):
                for s in msg["medquad_sources"]:
                    st.markdown(
                        f"**{s['focus']}** ({s['qtype']}) — *{s['source']}*\n\n"
                        f"> {s['snippet'][:200]}…"
                    )


# Send handlers

def handle_send(question: str, uploaded_image=None, is_clarification_reply: bool = False):
    """Customer service pipeline."""
    if not question.strip():
        return

    image_bytes = None
    image_name  = None
    if uploaded_image is not None:
        image_bytes = uploaded_image.read()
        image_name  = uploaded_image.name

    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.total_queries += 1
    if image_bytes:
        st.session_state.image_queries += 1

    with st.spinner("🔄 Analysing… (ambiguity → vision → retrieval → reasoning → validation)"):
        result = get_multimodal_response(
            question=question,
            conversation_history=st.session_state.chat_history[:-1],
            image_bytes=image_bytes,
            skip_ambiguity=is_clarification_reply,
        )

    if result.get("is_ambiguous"):
        st.session_state.pending_clarifications = result.get("clarifications", [])
        st.session_state.pending_question = question
        st.session_state.chat_history.pop()
        st.rerun()
        return

    assistant_msg = {
        "role":              "assistant",
        "content":           result["answer"],
        "confidence":        result["confidence"],
        "reasoning_steps":   result["reasoning_steps"],
        "sources_used":      result["sources_used"],
        "visual_tags":       result["visual_tags"],
        "is_valid":          result["is_valid"],
        "validation_issues": result["validation_issues"],
        "context_reference": result["context_reference"],
        "sentiment":         result.get("sentiment", "neutral"),
        "sentiment_score":   result.get("sentiment_score", 0.0),
        "emotion":           result.get("emotion", "neutral"),
        "escalate":          result.get("escalate", False),
        "detected_lang":      result.get("detected_lang", "en"),
        "detected_lang_name": result.get("detected_lang_name", "English"),
        "detected_lang_flag": result.get("detected_lang_flag", "🇬🇧"),
        "lang_confidence":    result.get("lang_confidence", 1.0),
        "is_mixed_language":  result.get("is_mixed_language", False),
        "english_question":   result.get("english_question", ""),
    }
    st.session_state.chat_history.append(assistant_msg)

    n = st.session_state.total_queries
    st.session_state.confidence_sum  += result["confidence"]
    st.session_state.avg_confidence   = st.session_state.confidence_sum / n

    detected_sentiment = result.get("sentiment", "neutral")
    if detected_sentiment in st.session_state.sentiment_counts:
        st.session_state.sentiment_counts[detected_sentiment] += 1
    st.session_state.sentiment_history.append(detected_sentiment)
    if result.get("escalate"):
        st.session_state.escalation_flag = True

    if image_bytes and image_name:
        added = update_knowledge_base(image_bytes_list=[(image_bytes, image_name)])
        if added:
            st.session_state.update_log.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] 🖼️ Image '{image_name}' ingested ({added} chunks)"
            )

    st.session_state.pending_clarifications = []
    st.session_state.pending_question = ""


def handle_medical_send(question: str):
    """Medical Q&A pipeline — calls get_medical_response."""
    if not question.strip():
        return

    st.session_state.med_chat_history.append({"role": "user", "content": question})
    st.session_state.med_total_queries += 1

    with st.spinner("⚕️ Querying MedQuAD… (NER → retrieval → reasoning)"):
        result = get_medical_response(
            question=question,
            conversation_history=st.session_state.med_chat_history[:-1],
        )

    assistant_msg = {
        "role":              "assistant",
        "content":           result["answer"],
        "confidence":        result["confidence"],
        "reasoning_steps":   result["reasoning_steps"],
        "sources_used":      result["sources_used"],
        "context_reference": result["context_reference"],
        "medquad_sources":   result.get("medquad_sources", []),
    }
    st.session_state.med_chat_history.append(assistant_msg)

    st.session_state.med_last_entities = result.get("medical_entities", {})
    st.session_state.med_last_sources  = result.get("medquad_sources", [])

    n = st.session_state.med_total_queries
    st.session_state.med_confidence_sum  += result["confidence"]
    st.session_state.med_avg_confidence   = st.session_state.med_confidence_sum / n


# Page header

st.title("🤖 AI Assistant Chatbot")
st.caption(
    "Powered by Gemini 2.0 Flash · RAG · Vision · NER · Multilingual"
)

# Tabs

tab_cs, tab_med, tab_arxiv, tab_kb = st.tabs([
    "💬 AI Chatbot",
    "🏥 Medical Q&A",
    "🔬 arXiv Expert",
    "🧠 Knowledge Base Manager",
])


# Tab 1 — AI Chatbot

with tab_cs:
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("💬 AI Chatbot")

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(f"🧑 {msg['content']}")
            else:
                render_assistant_message(msg)

        if st.session_state.pending_clarifications:
            st.warning(
                f"⚠️ Your question **\"{st.session_state.pending_question}\"** "
                "could be interpreted in multiple ways. Please clarify:"
            )
            cols = st.columns(len(st.session_state.pending_clarifications))
            for i, clarification in enumerate(st.session_state.pending_clarifications):
                with cols[i]:
                    if st.button(clarification, key=f"clarify_{i}"):
                        handle_send(clarification, is_clarification_reply=True)
                        st.rerun()

        st.markdown("**📎 Attach an image (optional):**")
        uploaded_image = st.file_uploader(
            "Upload image for visual analysis",
            type=["jpg", "jpeg", "png", "webp", "gif"],
            label_visibility="collapsed",
            key="cs_image_uploader",
        )
        if uploaded_image:
            st.image(uploaded_image, caption="Image ready to send", width=220)

        question = st.chat_input(
            "Ask something… (attach an image above for visual Q&A)",
            key="cs_chat_input",
        )
        if question:
            was_pending_clarification = bool(st.session_state.pending_clarifications)
            handle_send(question, uploaded_image=uploaded_image, is_clarification_reply=was_pending_clarification)
            st.rerun()

        if st.button("🗑️ Clear Chat", key="cs_clear"):
            for k in ["chat_history", "pending_clarifications", "pending_question"]:
                st.session_state[k] = [] if k != "pending_question" else ""
            for k in ["total_queries", "image_queries", "confidence_sum", "avg_confidence"]:
                st.session_state[k] = 0
            st.session_state.sentiment_counts  = {"positive": 0, "neutral": 0, "negative": 0}
            st.session_state.sentiment_history = []
            st.session_state.escalation_flag   = False
            st.rerun()

    with right_col:
        st.subheader("📊 Session Analytics")
        a1, a2, a3 = st.columns(3)
        a1.metric("Queries",     st.session_state.total_queries)
        a2.metric("With Images", st.session_state.image_queries)
        a3.metric(
            "Avg Confidence",
            f"{st.session_state.avg_confidence:.0%}" if st.session_state.total_queries else "—",
        )

        st.divider()
        st.subheader("😊 Customer Sentiment")
        counts = st.session_state.sentiment_counts
        total_sentiment = sum(counts.values())
        s1, s2, s3 = st.columns(3)
        s1.metric("😊 Positive", counts["positive"])
        s2.metric("😐 Neutral",  counts["neutral"])
        s3.metric("😟 Negative", counts["negative"])

        if total_sentiment:
            st.progress(
                counts["positive"] / total_sentiment,
                text=f"{counts['positive']}/{total_sentiment} positive interactions",
            )
        if st.session_state.escalation_flag:
            st.error(
                "🚨 At least one message in this session was flagged for possible "
                "human escalation — review the conversation."
            )
        else:
            st.caption("No escalation triggers detected yet.")


# Tab 2 — Medical Q&A

with tab_med:
    med_left, med_right = st.columns([2, 1])

    with med_left:
        st.subheader("🏥 Medical Q&A Chat")
        st.caption(
            "Powered by MedQuAD — 47,457 NIH-sourced Q&A pairs. "
            "⚕️ For informational purposes only. Always consult a healthcare professional."
        )

        for msg in st.session_state.med_chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(f"🧑 {msg['content']}")
            else:
                render_medical_message(msg)

        med_question = st.chat_input(
            "Ask a medical question… e.g. 'What are the symptoms of diabetes?'",
            key="med_chat_input",
        )
        if med_question:
            handle_medical_send(med_question)
            st.rerun()

        if st.button("🗑️ Clear Medical Chat", key="med_clear"):
            st.session_state.med_chat_history    = []
            st.session_state.med_total_queries   = 0
            st.session_state.med_confidence_sum  = 0.0
            st.session_state.med_avg_confidence  = 0.0
            st.session_state.med_last_entities   = {}
            st.session_state.med_last_sources    = []
            st.rerun()

    with med_right:
        st.subheader("📊 Medical Session Stats")
        m1, m2 = st.columns(2)
        m1.metric("Queries", st.session_state.med_total_queries)
        m2.metric(
            "Avg Confidence",
            f"{st.session_state.med_avg_confidence:.0%}"
            if st.session_state.med_total_queries else "—",
        )
        st.divider()

        st.subheader("🏷️ Detected Medical Entities")
        if st.session_state.med_last_entities:
            ner_display = format_entities_for_display(st.session_state.med_last_entities)
            st.markdown(ner_display)
        else:
            st.caption("Ask a medical question to see detected entities here.")
        st.divider()

        st.subheader("📖 MedQuAD Sources")
        if st.session_state.med_last_sources:
            for s in st.session_state.med_last_sources[:3]:
                with st.expander(f"📄 {s['focus']} ({s['qtype']})", expanded=False):
                    st.markdown(f"**Dataset:** {s['source']}")
                    st.markdown(f"**Snippet:** {s['snippet'][:200]}…")
        else:
            st.caption("Sources from the last MedQuAD retrieval will appear here.")
        st.divider()

        st.subheader("💡 Sample Questions")
        sample_questions = [
            "What are the symptoms of type 2 diabetes?",
            "How is hypertension treated?",
            "What causes migraines?",
            "What are the side effects of ibuprofen?",
            "How is pneumonia diagnosed?",
            "What is the treatment for asthma?",
        ]
        for sq in sample_questions:
            if st.button(sq, key=f"sq_{sq[:20]}"):
                handle_medical_send(sq)
                st.rerun()


# Tab 3 — arXiv Expert

with tab_arxiv:
    render_arxiv_tab()


# Tab 4 — Knowledge Base Manager

with tab_kb:
    kb_cs_col, kb_med_col, kb_arxiv_col = st.columns(3)

    # Customer Service KB

    with kb_cs_col:
        st.subheader("🤖 Customer Service KB")

        status_color = "🟢" if st.session_state.scheduler_running else "🔴"
        st.markdown(
            f"**Scheduler:** {status_color} "
            f"{'Running' if st.session_state.scheduler_running else 'Stopped'}"
        )
        st.markdown(f"**Last update:** `{st.session_state.last_update}`")
        st.divider()

        st.markdown("**Initial Setup**")
        if st.button("📦 Build KB from CSV"):
            with st.spinner("Ingesting base dataset…"):
                added = create_vector_db()
            st.success(f"✅ Knowledge base built — {added} chunks added!")

        st.divider()
        st.markdown("**🖼️ Ingest Images into KB**")
        kb_images = st.file_uploader(
            "Upload images to add to the knowledge base",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="kb_image_uploader",
            label_visibility="collapsed",
        )
        if st.button("📥 Ingest Images into KB") and kb_images:
            with st.spinner(f"Extracting text from {len(kb_images)} image(s)…"):
                image_bytes_list = [(img.read(), img.name) for img in kb_images]
                added = update_knowledge_base(image_bytes_list=image_bytes_list)
            st.success(f"✅ {added} image chunks added!")

        st.divider()
        st.markdown("**Web Sources**")
        web_input = st.text_area(
            "Enter URLs (one per line):",
            placeholder="https://yoursite.com/faq",
            height=80,
            key="cs_web_input",
        )
        web_urls = [u.strip() for u in web_input.splitlines() if u.strip()]

        st.markdown("**Extra CSV Sources**")
        csv_input = st.text_area(
            "Enter CSV file paths (one per line):",
            placeholder=r"c:\data\new_faqs.csv",
            height=60,
            key="cs_csv_input",
        )
        extra_csvs = [p.strip() for p in csv_input.splitlines() if p.strip()]

        st.divider()
        if st.button("🔄 Update CS KB Now"):
            with st.spinner("Updating knowledge base…"):
                run_update(web_urls=web_urls or None, extra_csvs=extra_csvs or None)
            st.success(f"✅ Done! Last update: {st.session_state.last_update}")

        st.divider()
        st.markdown("**Auto-Update Scheduler**")
        interval = st.slider("Update every (hours):", min_value=1, max_value=48, value=6)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start"):
                start_scheduler(interval, web_urls or None, extra_csvs or None)
                st.success(f"Scheduler started — every {interval}h")
                st.rerun()
        with col2:
            if st.button("⏹️ Stop"):
                stop_scheduler()
                st.info("Scheduler stopped.")
                st.rerun()

        st.divider()
        st.markdown("**Update Log**")
        if st.session_state.update_log:
            for entry in reversed(st.session_state.update_log[-10:]):
                st.caption(entry)
        else:
            st.caption("No updates yet.")

    # Medical KB

    with kb_med_col:
        st.subheader("🏥 Medical KB (MedQuAD)")

        med_index_exists = os.path.exists(MEDICAL_FAISS_PATH)
        st.markdown(
            f"**Index status:** {'🟢 Built' if med_index_exists else '🔴 Not built yet'}"
        )
        if med_index_exists and os.path.exists(MEDICAL_META_PATH):
            with open(MEDICAL_META_PATH) as f:
                meta = json.load(f)
            st.markdown(
                f"**Chunks:** {meta.get('total_chunks', '?')} &nbsp;|&nbsp; "
                f"**QA pairs:** {meta.get('total_qa_pairs', '?')}"
            )

        st.divider()
        st.markdown("**Build Medical Index from MedQuAD**")
        st.info(
            "1. Clone MedQuAD: `git clone https://github.com/abachaa/MedQuAD`\n"
            "2. Set the path below and click Build."
        )

        medquad_dir = st.text_input(
            "MedQuAD dataset directory:",
            value="",
            placeholder="e.g. /home/user/MedQuAD  or  C:\\data\\MedQuAD",
            key="medquad_dir",
        )
        max_files_input = st.number_input(
            "Max XML files to parse (0 = all ~900):",
            min_value=0, max_value=900, value=50, step=10,
            help="Start with 50 for a quick test, 0 for the full dataset.",
        )
        max_files = int(max_files_input) if max_files_input > 0 else None

        if st.button("🏗️ Build Medical Index"):
            with st.spinner(
                f"Parsing {'all' if not max_files else max_files} MedQuAD XML files…"
            ):
                try:
                    added = build_medical_index(dataset_dir=medquad_dir, max_files=max_files)
                    st.success(f"✅ Medical index built — {added} chunks indexed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Build failed: {e}")

        st.divider()
        st.markdown("**ℹ️ About MedQuAD**")
        st.markdown(
            "- **47,457 Q&A pairs** from 12 NIH websites\n"
            "- License: Creative Commons Attribution 4.0 (CC BY)\n"
            "- [GitHub Repository](https://github.com/abachaa/MedQuAD)"
        )

    # arXiv KB

    with kb_arxiv_col:
        st.subheader("🔬 arXiv Expert KB")

        arxiv_index_exists = os.path.exists(ARXIV_FAISS_PATH)
        st.markdown(
            f"**Index status:** {'🟢 Built' if arxiv_index_exists else '🔴 Not built yet'}"
        )
        if arxiv_index_exists and os.path.exists(ARXIV_META_PATH):
            with open(ARXIV_META_PATH) as f:
                arxiv_meta = json.load(f)
            st.markdown(
                f"**Papers:** {arxiv_meta.get('total_papers', '?')} &nbsp;|&nbsp; "
                f"**Chunks:** {arxiv_meta.get('total_chunks', '?')}"
            )

        st.divider()
        st.markdown("**Build arXiv Index**")
        st.info(
            "1. Download the arXiv dataset from Kaggle:\n"
            "   [Cornell-University/arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv)\n"
            "2. Extract `arxiv-metadata-oai-snapshot.json`\n"
            "3. Set the path below and click Build."
        )

        arxiv_json_path = st.text_input(
            "Path to arxiv-metadata-oai-snapshot.json:",
            value="",
            placeholder="e.g. /home/user/arxiv-metadata-oai-snapshot.json",
            key="arxiv_json_path",
        )
        max_papers_input = st.number_input(
            "Max ML papers to index (0 = all, may be slow):",
            min_value=0, max_value=100000, value=5000, step=1000,
            help="5,000 papers is a good starting point (~2-3 min build time).",
        )
        max_papers = int(max_papers_input) if max_papers_input > 0 else None

        if st.button("🏗️ Build arXiv Index"):
            if not os.path.exists(arxiv_json_path):
                st.error(f"❌ File not found: {arxiv_json_path}")
            else:
                with st.spinner(
                    f"Loading {'all' if not max_papers else max_papers} ML papers "
                    "and building FAISS index… (this may take a few minutes)"
                ):
                    try:
                        added = build_arxiv_index(
                            json_path=arxiv_json_path,
                            max_papers=max_papers or 5000,
                        )
                        st.success(f"✅ arXiv index built — {added} chunks indexed!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Build failed: {e}")

        st.divider()
        st.markdown("**ℹ️ About arXiv ML Dataset**")
        st.markdown(
            "- **1.7M+ papers** (full snapshot); ML subset: ~200K\n"
            "- Categories: cs.LG · stat.ML · cs.AI · cs.CV · cs.CL · cs.NE · cs.IR\n"
            "- Updated monthly by Cornell University\n"
            "- [Kaggle Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)\n"
            "- License: Creative Commons CC0"
        )
