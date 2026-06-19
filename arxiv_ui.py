import os
import json
import tempfile
import streamlit as st
from pathlib import Path

from arxiv_helper import (
    build_arxiv_index,
    load_arxiv_dataset,
    retrieve_arxiv_docs,
    summarise_paper,
    explain_concept,
    get_arxiv_response,
    build_concept_graph,
    extract_ml_entities,
    format_ml_entities_for_display,
    ARXIV_FAISS_PATH,
    ARXIV_META_PATH,
)


# Session state defaults

ARXIV_DEFAULTS = {
    "arxiv_chat_history":    [],
    "arxiv_total_queries":   0,
    "arxiv_confidence_sum":  0.0,
    "arxiv_avg_confidence":  0.0,
    "arxiv_last_docs":       [],
    "arxiv_search_results":  [],
    "arxiv_search_query":    "",
    "arxiv_concept_result":  None,
    "arxiv_concept_input":   "",
    "arxiv_last_entities":   {},
}


def init_arxiv_session_state():
    """Initialise all arXiv-related session state keys."""
    for k, v in ARXIV_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


# Helpers

def confidence_badge(score: float) -> str:
    if score >= 0.75:
        return f"🟢 High confidence ({score:.0%})"
    elif score >= 0.45:
        return f"🟡 Medium confidence ({score:.0%})"
    else:
        return f"🔴 Low confidence ({score:.0%})"


def render_paper_card(paper: dict, idx: int):
    """Render a single paper result card with summarise + arXiv link."""
    arxiv_id = paper.get("arxiv_id", "")
    title    = paper.get("title", "Unknown Title")
    authors  = paper.get("authors", "")
    cats     = paper.get("categories", "")
    year     = paper.get("year", "")
    snippet  = paper.get("snippet", "")

    with st.container(border=True):
        st.markdown(f"**{idx}. {title}** ({year})")
        col_info, col_btns = st.columns([3, 1])
        with col_info:
            if authors:
                st.caption(f"👤 {authors[:90]}{'…' if len(authors) > 90 else ''}")
            st.caption(f"🏷️ {cats}  |  📄 arXiv:{arxiv_id}")
            st.markdown(f"> {snippet[:260]}…")
        with col_btns:
            if arxiv_id:
                st.link_button("🔗 arXiv", f"https://arxiv.org/abs/{arxiv_id}", use_container_width=True)
            if st.button("📝 Summarise", key=f"sum_{arxiv_id}_{idx}", use_container_width=True):
                abstract = snippet
                if "Abstract:" in abstract:
                    abstract = abstract.split("Abstract:", 1)[1].strip()
                with st.spinner("Generating summary…"):
                    summary = summarise_paper(title, abstract)
                st.info(f"**Summary:** {summary}")


def render_concept_explainer():
    """Concept Explainer sub-section."""
    st.subheader("💡 ML Concept Explainer")
    st.caption("Enter any ML concept for a plain-English explanation with key points, analogy, and related terms.")

    concept_input = st.text_input(
        "Concept to explain",
        placeholder="e.g. attention mechanism, dropout, federated learning, UMAP…",
        key="arxiv_concept_input_field",
        label_visibility="collapsed",
    )

    if st.button("🔍 Explain Concept", key="explain_btn") and concept_input.strip():
        with st.spinner(f"Retrieving arXiv context for '{concept_input}'…"):
            docs = retrieve_arxiv_docs(concept_input.strip(), k=3)
            retrieved_context = "\n\n".join(d.page_content[:400] for d in docs)
        with st.spinner("Generating explanation…"):
            result = explain_concept(concept_input.strip(), retrieved_context=retrieved_context)
        st.session_state.arxiv_concept_result = result
        st.session_state.arxiv_concept_input  = concept_input.strip()

    result = st.session_state.get("arxiv_concept_result")
    if result:
        concept = st.session_state.get("arxiv_concept_input", "")
        st.markdown(f"### 📖 {concept.title()}")

        st.markdown(result.get("explanation", ""))

        col_left, col_right = st.columns(2)
        with col_left:
            key_points = result.get("key_points", [])
            if key_points:
                st.markdown("**🔑 Key Points:**")
                for kp in key_points:
                    st.markdown(f"- {kp}")
        with col_right:
            related = result.get("related_terms", [])
            if related:
                st.markdown("**🔗 Related Terms:**")
                st.markdown(" · ".join(f"`{r}`" for r in related))

        example = result.get("example", "")
        if example:
            st.info(f"💡 **Example / Analogy:** {example}")

        papers = result.get("papers", [])
        if papers:
            st.markdown("**📚 Influential Papers:**")
            for p in papers:
                st.markdown(f"- {p}")


def render_topic_graph(docs: list):
    """
    Render a pyvis topic co-occurrence graph from retrieved LangChain Documents.
    Falls back to a text summary if pyvis is not installed.
    """
    st.subheader("🕸️ Concept Co-occurrence Graph")
    st.caption(
        "ML concepts that frequently appear together in the retrieved papers. "
        "Node size = frequency · Edge thickness = co-occurrence strength."
    )

    if not docs:
        st.info("Run a paper search or ask a question first to generate the concept graph.")
        return

    graph_data = build_concept_graph(docs, top_n=18)
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    if not nodes:
        st.warning("Not enough concept data to build a graph from these results.")
        return

    try:
        from pyvis.network import Network
        import streamlit.components.v1 as components

        net = Network(
            height="480px",
            width="100%",
            bgcolor="#0e1117",
            font_color="white",
            notebook=False,
        )
        net.set_options("""{
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -55,
              "centralGravity": 0.01,
              "springLength": 130
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 120}
          },
          "interaction": {"hover": true}
        }""")

        max_count = max((n["count"] for n in nodes), default=1)
        for node in nodes:
            size = 10 + int((node["count"] / max_count) * 40)
            net.add_node(
                node["id"],
                label=node["id"],
                size=size,
                title=f"{node['id']}: {node['count']} papers",
                color="#4A90E2",
            )

        if edges:
            max_w = max((e["weight"] for e in edges), default=1)
            for edge in edges:
                width = 1 + int((edge["weight"] / max_w) * 8)
                net.add_edge(
                    edge["source"],
                    edge["target"],
                    value=width,
                    title=f"Co-occurs {edge['weight']} times",
                )

        with tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            net.save_graph(tmp.name)
            html_content = Path(tmp.name).read_text(encoding="utf-8")

        components.html(html_content, height=500, scrolling=False)

    except ImportError:
        st.warning("Install `pyvis` for an interactive graph: `pip install pyvis`")
        st.markdown("**Top concepts by frequency:**")
        for n in sorted(nodes, key=lambda x: x["count"], reverse=True)[:15]:
            bar = "█" * min(n["count"] // 2 + 1, 20)
            st.text(f"{n['id']:35s} {bar} ({n['count']})")
        if edges:
            st.markdown("**Strongest co-occurrences:**")
            for e in sorted(edges, key=lambda x: x["weight"], reverse=True)[:8]:
                st.text(f"  {e['source']} ↔ {e['target']}  ({e['weight']}x)")


# Main tab renderer

def render_arxiv_tab():
    """
    Full arXiv Expert tab. Call inside `with tab_arxiv:` in main.py.
    """
    arxiv_left, arxiv_right = st.columns([2, 1])

    # Left — Paper Search · Concept Explainer · Expert Chat

    with arxiv_left:

        index_exists = os.path.exists(ARXIV_FAISS_PATH)
        if not index_exists:
            st.warning(
                "⚠️ arXiv index not built yet. "
                "Go to **🧠 Knowledge Base Manager → arXiv Expert KB** to build it."
            )

        # Section 1: Paper Search

        st.subheader("🔍 Paper Search")
        st.caption("Semantic search over arXiv ML papers (cs.LG · stat.ML · cs.AI · cs.CV · cs.CL · cs.NE)")

        search_col, btn_col = st.columns([4, 1])
        with search_col:
            search_query = st.text_input(
                "Search",
                placeholder="e.g. transformers for image classification, federated privacy…",
                key="arxiv_search_field",
                label_visibility="collapsed",
            )
        with btn_col:
            search_btn = st.button("🔎 Search", key="arxiv_search_btn", use_container_width=True)

        if search_btn and search_query.strip():
            if not index_exists:
                st.error("Build the arXiv index first via Knowledge Base Manager.")
            else:
                with st.spinner("Searching arXiv FAISS index…"):
                    docs = retrieve_arxiv_docs(search_query.strip(), k=6)
                    results = []
                    for d in docs:
                        m = d.metadata
                        results.append({
                            "title":      m.get("title", ""),
                            "arxiv_id":   m.get("arxiv_id", ""),
                            "authors":    m.get("authors", ""),
                            "categories": m.get("categories", ""),
                            "year":       m.get("year", ""),
                            "snippet":    d.page_content[:300],
                        })
                st.session_state.arxiv_search_results = results
                st.session_state.arxiv_search_query   = search_query.strip()
                st.session_state.arxiv_last_docs      = docs

        results = st.session_state.get("arxiv_search_results", [])
        if results:
            st.markdown(
                f"**{len(results)} results for:** *{st.session_state.get('arxiv_search_query', '')}*"
            )
            for i, paper in enumerate(results, 1):
                render_paper_card(paper, i)
        elif search_btn and search_query.strip():
            st.info("No results found. Try a different query or build the index with more papers.")

        st.divider()

        # Section 2: Concept Explainer

        render_concept_explainer()

        st.divider()

        # Section 3: Expert Chat

        st.subheader("🤖 arXiv Expert Chat")
        st.caption(
            "Ask complex ML questions, request paper comparisons, or follow up on search results. "
            "Powered by Gemini + arXiv RAG."
        )

        for msg in st.session_state.arxiv_chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(f"🧑 {msg['content']}")
            else:
                with st.chat_message("assistant"):
                    st.markdown(f"🔬 {msg['content']}")
                    meta_cols = st.columns(3)
                    with meta_cols[0]:
                        st.caption(confidence_badge(msg.get("confidence", 0.6)))
                    with meta_cols[1]:
                        ents = msg.get("ml_entities", {})
                        methods = ents.get("methods", [])
                        if methods:
                            st.caption(f"🔧 {', '.join(methods[:3])}")
                    with meta_cols[2]:
                        if msg.get("context_reference"):
                            st.caption(f"🔗 {msg['context_reference']}")

                    if msg.get("reasoning_steps"):
                        with st.expander("🧠 Reasoning trace", expanded=False):
                            for i, step in enumerate(msg["reasoning_steps"], 1):
                                st.write(f"**Step {i}:** {step}")

                    arxiv_sources = msg.get("arxiv_sources", [])
                    if arxiv_sources:
                        with st.expander(f"📄 {len(arxiv_sources)} arXiv papers used", expanded=False):
                            for s in arxiv_sources:
                                st.markdown(
                                    f"**[arXiv:{s['arxiv_id']}]** {s['title']} ({s['year']})\n\n"
                                    f"> {s['snippet'][:200]}…"
                                )

                    summaries = msg.get("paper_summaries", [])
                    if summaries:
                        with st.expander("📝 Auto-generated paper summaries", expanded=False):
                            for ps in summaries:
                                st.markdown(f"**{ps['title']}** ({ps['year']})")
                                st.markdown(ps["summary"])
                                st.divider()

        arxiv_question = st.chat_input(
            "Ask about ML research… e.g. 'How do diffusion models compare to GANs?'",
            key="arxiv_chat_input",
        )
        if arxiv_question:
            st.session_state.arxiv_chat_history.append(
                {"role": "user", "content": arxiv_question}
            )
            st.session_state.arxiv_total_queries += 1

            with st.spinner("🔬 Retrieving arXiv papers and generating expert answer…"):
                result = get_arxiv_response(
                    question=arxiv_question,
                    conversation_history=st.session_state.arxiv_chat_history[:-1],
                )

            assistant_msg = {
                "role":             "assistant",
                "content":          result["answer"],
                "confidence":       result["confidence"],
                "reasoning_steps":  result["reasoning_steps"],
                "sources_used":     result["sources_used"],
                "context_reference":result["context_reference"],
                "ml_entities":      result["ml_entities"],
                "arxiv_sources":    result["arxiv_sources"],
                "paper_summaries":  result["paper_summaries"],
            }
            st.session_state.arxiv_chat_history.append(assistant_msg)

            n = st.session_state.arxiv_total_queries
            st.session_state.arxiv_confidence_sum += result["confidence"]
            st.session_state.arxiv_avg_confidence  = st.session_state.arxiv_confidence_sum / n
            st.session_state.arxiv_last_entities   = result["ml_entities"]

            if result.get("arxiv_sources"):
                from langchain_core.documents import Document as LCDoc
                mock_docs = [
                    LCDoc(
                        page_content=f"Title: {s['title']}\n\nAbstract: {s['snippet']}",
                        metadata=s,
                    )
                    for s in result["arxiv_sources"]
                ]
                st.session_state.arxiv_last_docs = mock_docs

            st.rerun()

        if st.button("🗑️ Clear arXiv Chat", key="arxiv_clear"):
            st.session_state.arxiv_chat_history   = []
            st.session_state.arxiv_total_queries  = 0
            st.session_state.arxiv_confidence_sum = 0.0
            st.session_state.arxiv_avg_confidence = 0.0
            st.session_state.arxiv_last_entities  = {}
            st.rerun()

    # Right — Stats · ML Entities · Topic Graph · Sample Queries

    with arxiv_right:

        st.subheader("📊 Session Stats")
        r1, r2 = st.columns(2)
        r1.metric("Queries", st.session_state.arxiv_total_queries)
        r2.metric(
            "Avg Confidence",
            f"{st.session_state.arxiv_avg_confidence:.0%}"
            if st.session_state.arxiv_total_queries else "—",
        )
        st.divider()

        st.subheader("🏷️ Detected ML Entities")
        last_ents = st.session_state.get("arxiv_last_entities", {})
        if last_ents and any(last_ents.values()):
            st.markdown(format_ml_entities_for_display(last_ents))
        else:
            st.caption("ML entities from the latest query appear here.")
        st.divider()

        render_topic_graph(st.session_state.get("arxiv_last_docs", []))
        st.divider()

        st.subheader("💡 Sample Questions")
        samples = [
            "What is the attention mechanism in transformers?",
            "How does federated learning preserve privacy?",
            "Explain the difference between GANs and diffusion models",
            "What are recent advances in few-shot learning?",
            "How does BERT handle contextual word embeddings?",
            "What is knowledge distillation used for?",
            "Explain gradient descent and its variants",
            "What is contrastive learning?",
        ]
        for sq in samples:
            if st.button(sq, key=f"arxiv_sq_{sq[:28]}"):
                st.session_state.arxiv_chat_history.append({"role": "user", "content": sq})
                st.session_state.arxiv_total_queries += 1
                with st.spinner("🔬 Thinking…"):
                    result = get_arxiv_response(
                        question=sq,
                        conversation_history=st.session_state.arxiv_chat_history[:-1],
                    )
                assistant_msg = {
                    "role": "assistant", "content": result["answer"],
                    "confidence": result["confidence"],
                    "reasoning_steps": result["reasoning_steps"],
                    "sources_used": result["sources_used"],
                    "context_reference": result["context_reference"],
                    "ml_entities": result["ml_entities"],
                    "arxiv_sources": result["arxiv_sources"],
                    "paper_summaries": result["paper_summaries"],
                }
                st.session_state.arxiv_chat_history.append(assistant_msg)
                n = st.session_state.arxiv_total_queries
                st.session_state.arxiv_confidence_sum += result["confidence"]
                st.session_state.arxiv_avg_confidence  = st.session_state.arxiv_confidence_sum / n
                st.session_state.arxiv_last_entities   = result["ml_entities"]
                if result.get("arxiv_sources"):
                    from langchain_core.documents import Document as LCDoc
                    st.session_state.arxiv_last_docs = [
                        LCDoc(
                            page_content=f"Title: {s['title']}\n\nAbstract: {s['snippet']}",
                            metadata=s,
                        )
                        for s in result["arxiv_sources"]
                    ]
                st.rerun()
