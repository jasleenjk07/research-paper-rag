import json
from pathlib import Path

import streamlit as st

from src.config import (
    EMBEDDING_MODEL,
    FINAL_TOP_K,
    HYBRID_RETRIEVE_K,
    LLM_MODEL,
    RERANKER_MODEL,
    PROCESSED_PAPERS_PATH,
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
)

from src.generator import get_paper_title


RESULTS_PATH = Path(
    "evaluation/results.json"
)

QUESTIONS_PATH = Path(
    "evaluation/questions.json"
)

UPLOADS_DIR = Path(
    "uploads"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Research Paper RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700'
        '&family=IBM+Plex+Mono:wght@400;500&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                ellipse at top left,
                #e8f0ea 0%,
                transparent 45%
            ),
            radial-gradient(
                ellipse at bottom right,
                #e6ebe4 0%,
                transparent 40%
            ),
            #f7f6f2;
    }

    [data-testid="stSidebar"] {
        background: #1c2420;
    }

    [data-testid="stSidebar"] * {
        color: #e8ebe6 !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        font-weight: 500;
    }

    h1, h2, h3 {
        color: #1c2420 !important;
        letter-spacing: -0.02em;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1c2420;
        margin-bottom: 0.2rem;
    }

    .hero-sub {
        color: #5a655c;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #d8ddd6;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        box-shadow: 0 1px 2px rgba(28, 36, 32, 0.04);
    }

    .metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6b756e;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.15rem;
        font-weight: 500;
        color: #1c2420;
        word-break: break-all;
    }

    .answer-box {
        background: #ffffff;
        border-left: 4px solid #2f6b4f;
        border-radius: 0 12px 12px 0;
        padding: 1.25rem 1.4rem;
        margin: 0.5rem 0 1.2rem 0;
        box-shadow: 0 1px 3px rgba(28, 36, 32, 0.06);
        line-height: 1.65;
        color: #1c2420;
    }

    .source-preview {
        color: #3a433d;
        font-size: 0.92rem;
        line-height: 1.55;
        white-space: pre-wrap;
    }

    .paper-chip {
        display: inline-block;
        background: #eef3ef;
        border: 1px solid #c9d4cb;
        color: #1c2420;
        border-radius: 8px;
        padding: 0.45rem 0.75rem;
        margin: 0.25rem 0.35rem 0.25rem 0;
        font-size: 0.88rem;
    }

    div[data-testid="stButton"] > button {
        background: #2f6b4f;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
    }

    div[data-testid="stButton"] > button:hover {
        background: #255640;
        color: white;
        border: none;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PIPELINE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_pipeline():

    from src.rag_pipeline import RAGPipeline

    return RAGPipeline()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_json(path: Path):

    if not path.exists():
        return None

    with open(
        path,
        "r",
    ) as file:

        return json.load(file)


def vector_store_ready():

    return (
        Path(
            FAISS_INDEX_PATH
        ).exists()
        and Path(
            CHUNKS_PATH
        ).exists()
    )


def get_processed_documents():

    processed = load_json(
        Path(
            PROCESSED_PAPERS_PATH
        )
    )

    if not processed:
        return []

    return processed


# ============================================================
# UPLOAD + INGESTION
# ============================================================

def process_uploaded_files(
    uploaded_files
):
    """
    Save uploaded PDFs and ingest them into the shared
    FAISS + BM25 knowledge base.

    Returns:
        list[str]: Successfully processed document names.
    """

    UPLOADS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_paths = []

    for uploaded_file in uploaded_files:

        safe_filename = Path(
            uploaded_file.name
        ).name

        file_path = (
            UPLOADS_DIR
            / safe_filename
        )

        with open(
            file_path,
            "wb",
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )

        saved_paths.append(
            file_path
        )

    if not saved_paths:
        return []

    from src.ingest import (
        ingest_uploaded_files
    )

    processed_files = (
        ingest_uploaded_files(
            saved_paths
        )
    )

    if processed_files is None:
        processed_files = []

    if isinstance(
        processed_files,
        str,
    ):

        processed_files = [
            processed_files
        ]

    if isinstance(
        processed_files,
        tuple,
    ):

        processed_files = list(
            processed_files
        )

    if isinstance(
        processed_files,
        set,
    ):

        processed_files = list(
            processed_files
        )

    if not isinstance(
        processed_files,
        list,
    ):

        processed_files = []

    load_pipeline.clear()

    return processed_files


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    st.sidebar.markdown(
        "## Research Paper RAG"
    )

    st.sidebar.caption(
        "Hybrid retrieval + reranking + local LLM"
    )

    page = st.sidebar.radio(
        "Navigate",
        [
            "Ask",
            "Papers",
            "Evaluation",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown(
        "---"
    )

    st.sidebar.markdown(
        "**Pipeline**"
    )

    st.sidebar.text(
        f"Embed: "
        f"{EMBEDDING_MODEL.split('/')[-1]}"
    )

    st.sidebar.text(
        f"LLM: {LLM_MODEL}"
    )

    st.sidebar.text(
        f"Rerank: "
        f"{RERANKER_MODEL.split('/')[-1]}"
    )

    st.sidebar.text(
        f"Retrieve K: "
        f"{HYBRID_RETRIEVE_K}"
    )

    st.sidebar.text(
        f"Final K: "
        f"{FINAL_TOP_K}"
    )

    return page


# ============================================================
# ASK PAGE
# ============================================================

def render_ask_page():

    st.markdown(
        '<div class="hero-title">'
        'Ask your documents'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-sub">'
        'Upload your own PDFs or use the built-in research papers. '
        'Select a document for focused questions, or search across '
        'all documents when needed.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ========================================================
    # UPLOAD
    # ========================================================

    st.markdown(
        "### 📄 Add your own documents"
    )

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help=(
            "Upload one or more PDF documents "
            "to add them to the knowledge base."
        ),
    )

    if uploaded_files:

        st.write(
            f"{len(uploaded_files)} "
            f"document(s) selected."
        )

        if st.button(
            "Process Documents",
            type="secondary",
        ):

            with st.spinner(
                "Loading PDF, creating chunks, "
                "generating embeddings, and updating "
                "the knowledge base..."
            ):

                try:

                    processed_files = (
                        process_uploaded_files(
                            uploaded_files
                        )
                    )

                    if processed_files:

                        # Automatically focus on the
                        # most recently processed document.
                        st.session_state[
                            "selected_document"
                        ] = processed_files[-1]

                        st.success(
                            f"Successfully processed "
                            f"{len(processed_files)} "
                            f"document(s)."
                        )

                        st.write(
                            "Processed:"
                        )

                        for filename in (
                            processed_files
                        ):

                            st.write(
                                f"• {filename}"
                            )

                        st.info(
                            "The RAG pipeline has been "
                            "refreshed and the newest "
                            "document is now selected."
                        )

                    else:

                        st.warning(
                            "No new documents were processed. "
                            "The uploaded document may already "
                            "exist in the knowledge base, or "
                            "the ingestion failed."
                        )

                except Exception as exc:

                    st.error(
                        f"Failed to process documents: "
                        f"{exc}"
                    )

    # ========================================================
    # VECTOR STORE CHECK
    # ========================================================

    if not vector_store_ready():

        st.error(
            "No document index found. "
            "Run the initial ingestion or add documents "
            "using the uploader."
        )

        return

    # ========================================================
    # DOCUMENT SELECTION
    # ========================================================

    processed_documents = (
        get_processed_documents()
    )

    if not processed_documents:

        st.error(
            "No processed documents are available."
        )

        return

    st.markdown(
        "### 📚 Document scope"
    )

    document_options = [
        "All documents"
    ] + processed_documents

    current_selection = (
        st.session_state.get(
            "selected_document",
            "All documents",
        )
    )

    if (
        current_selection
        not in document_options
    ):

        current_selection = (
            "All documents"
        )

    selected_document = st.selectbox(
        "Choose which document to search",
        document_options,
        index=document_options.index(
            current_selection
        ),
        key="document_selector",
        help=(
            "Choose a specific document for "
            "focused questions, or select "
            "'All documents' for cross-document "
            "questions."
        ),
    )

    st.session_state[
        "selected_document"
    ] = selected_document

    if selected_document == "All documents":

        st.caption(
            "Searching across all indexed documents."
        )

    else:

        st.caption(
            f"Searching only within: "
            f"{Path(selected_document).stem}"
        )

    # ========================================================
    # QUESTION
    # ========================================================

    st.markdown(
        "### 💬 Ask a question"
    )

    examples = [
        "Why can Transformers be trained in parallel?",
        "How does LoRA reduce trainable parameters?",
        "Why does BERT use masked language modeling?",
        "How does RAG use retrieved documents during generation?",
    ]

    cols = st.columns(
        len(examples)
    )

    for col, example in zip(
        cols,
        examples,
    ):

        if col.button(
            example,
            use_container_width=True,
            key=f"ex_{example}",
        ):

            st.session_state[
                "question_input"
            ] = example

    question = st.text_area(
        "Your question",
        key="question_input",
        height=100,
        placeholder=(
            "Ask something about "
            "the selected document..."
        ),
    )

    ask = st.button(
        "Generate answer",
        type="primary",
    )

    if not ask:
        return

    if not question or not question.strip():

        st.warning(
            "Enter a question first."
        )

        return

    # ========================================================
    # DETERMINE DOCUMENT FILTER
    # ========================================================

    if selected_document == "All documents":

        document_name = None

    else:

        document_name = selected_document

    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    with st.spinner(
        "Retrieving relevant documents "
        "and generating answer..."
    ):

        try:

            pipeline = load_pipeline()

            answer = pipeline.answer(
                question.strip(),
                document_name=document_name,
            )

            sources = (
                pipeline.last_retrieved_documents
            )

        except Exception as exc:

            st.error(
                f"Failed to generate answer: "
                f"{exc}"
            )

            return

    # ========================================================
    # ANSWER
    # ========================================================

    st.markdown(
        "### Answer"
    )

    st.markdown(
        '<div class="answer-box">'
        + answer
        + '</div>',
        unsafe_allow_html=True,
    )

    # ========================================================
    # RETRIEVED SOURCES
    # ========================================================

    st.markdown(
        "### Retrieved sources"
    )

    if not sources:

        st.info(
            "No sources retrieved."
        )

        return

    for rank, (
        document,
        score,
    ) in enumerate(
        sources,
        start=1,
    ):

        title = get_paper_title(
            document
        )

        page = document.metadata.get(
            "page",
            0,
        )

        page = page + 1

        chunk_id = document.metadata.get(
            "chunk_id",
            "?",
        )

        preview = (
            document.page_content
            .replace("\n", " ")
            .strip()
        )

        if len(preview) > 500:

            preview = (
                preview[:500]
                + "..."
            )

        with st.container(
            border=True
        ):

            st.markdown(
                f"### #{rank} — {title}"
            )

            st.caption(
                f"Page {page} · "
                f"Chunk {chunk_id} · "
                f"Score {score:.4f}"
            )

            st.write(
                preview
            )


# ============================================================
# PAPERS PAGE
# ============================================================

def render_papers_page():

    st.markdown(
        '<div class="hero-title">'
        'Indexed documents'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-sub">'
        'Documents currently available '
        'in the knowledge base.'
        '</div>',
        unsafe_allow_html=True,
    )

    processed = (
        get_processed_documents()
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Documents
                </div>

                <div class="metric-value">
                    {len(processed)}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Final top-k
                </div>

                <div class="metric-value">
                    {FINAL_TOP_K}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Hybrid retrieve-k
                </div>

                <div class="metric-value">
                    {HYBRID_RETRIEVE_K}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "### Documents"
    )

    if not processed:

        st.info(
            "No processed documents found. "
            "Add documents using the uploader "
            "or run the initial ingestion."
        )

        return

    for name in processed:

        st.markdown(
            f"""
            <span class="paper-chip">
                {Path(name).stem}
            </span>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "### Details"
    )

    for name in processed:

        st.write(
            f"- `{name}`"
        )


# ============================================================
# EVALUATION PAGE
# ============================================================

def render_evaluation_page():

    st.markdown(
        '<div class="hero-title">'
        'Evaluation results'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-sub">'
        'Browse saved outputs from '
        '`evaluation/run_evaluation.py`.'
        '</div>',
        unsafe_allow_html=True,
    )

    results = load_json(
        RESULTS_PATH
    )

    if not results:

        st.info(
            "No evaluation results found. "
            "Run `python -m evaluation.run_evaluation` "
            "first."
        )

        return

    categories = sorted(
        {
            item.get(
                "category",
                "unknown",
            )
            for item in results
        }
    )

    difficulties = sorted(
        {
            item.get(
                "difficulty",
                "unknown",
            )
            for item in results
        }
    )

    f1, f2 = st.columns(2)

    selected_category = f1.selectbox(
        "Category",
        ["All"] + categories,
    )

    selected_difficulty = f2.selectbox(
        "Difficulty",
        ["All"] + difficulties,
    )

    filtered = results

    if selected_category != "All":

        filtered = [
            item
            for item in filtered
            if item.get("category")
            == selected_category
        ]

    if selected_difficulty != "All":

        filtered = [
            item
            for item in filtered
            if item.get("difficulty")
            == selected_difficulty
        ]

    avg_time = (
        sum(
            item.get(
                "total_time"
            )
            or 0
            for item in filtered
        )
        / len(filtered)
        if filtered
        else 0
    )

    m1, m2, m3 = st.columns(3)

    with m1:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Questions
                </div>

                <div class="metric-value">
                    {len(filtered)}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Avg total time
                </div>

                <div class="metric-value">
                    {avg_time:.1f}s
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Dataset size
                </div>

                <div class="metric-value">
                    {len(results)}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    if not filtered:

        st.warning(
            "No results match "
            "the selected filters."
        )

        return

    labels = [
        f"{idx + 1}. "
        f"{item['question'][:80]}"
        for idx, item in enumerate(
            filtered
        )
    ]

    choice = st.selectbox(
        "Question",
        labels,
    )

    item = filtered[
        labels.index(choice)
    ]

    st.markdown(
        "### Question"
    )

    st.write(
        item["question"]
    )

    meta_cols = st.columns(4)

    meta_cols[0].write(
        f"**Category:** "
        f"{item.get('category', '-')}"
    )

    meta_cols[1].write(
        f"**Difficulty:** "
        f"{item.get('difficulty', '-')}"
    )

    meta_cols[2].write(
        f"**Expected paper:** "
        f"{item.get('expected_paper', '-')}"
    )

    meta_cols[3].write(
        f"**Time:** "
        f"{item.get('total_time', '-')}s"
    )

    st.markdown(
        "### Generated answer"
    )

    st.markdown(
        f'<div class="answer-box">'
        f'{item.get("generated_answer", "")}'
        f'</div>',
        unsafe_allow_html=True,
    )

    with st.expander(
        "Ground truth"
    ):

        st.write(
            item.get(
                "ground_truth",
                "",
            )
        )

    citations = (
        item.get("citations")
        or []
    )

    if citations:

        st.markdown(
            "### Citations"
        )

        for rank, citation in enumerate(
            citations,
            start=1,
        ):

            paper = (
                citation.get("paper")
                or "Unknown paper"
            )

            page = citation.get(
                "page",
                "?",
            )

            score = citation.get(
                "score",
                0,
            )

            st.write(
                f"{rank}. **{paper}** "
                f"· Page {page} "
                f"· Score {score:.3f}"
            )

    contexts = (
        item.get(
            "retrieved_contexts"
        )
        or []
    )

    if contexts:

        st.markdown(
            "### Retrieved contexts"
        )

        for rank, context in enumerate(
            contexts,
            start=1,
        ):

            with st.expander(
                f"Context {rank}"
            ):

                st.write(
                    context
                )


# ============================================================
# MAIN
# ============================================================

def main():

    page = render_sidebar()

    if page == "Ask":

        render_ask_page()

    elif page == "Papers":

        render_papers_page()

    else:

        render_evaluation_page()


if __name__ == "__main__":

    main()