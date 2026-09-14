import html
import json
import os
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


    /* ======================================================
       GLOBAL FONT
       ====================================================== */

    html,
    body {
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


    /* ======================================================
       SIDEBAR
       ====================================================== */

    [data-testid="stSidebar"] {
        background: #1c2420;
    }

    [data-testid="stSidebar"] * {
        color: #e8ebe6 !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        font-weight: 500;
    }


    /* ======================================================
       HEADINGS
       ====================================================== */

    h1,
    h2,
    h3 {
        color: #1c2420 !important;
        letter-spacing: -0.02em;
    }


    /* ======================================================
       HERO
       ====================================================== */

    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1c2420 !important;
        margin-bottom: 0.2rem;
    }

    .hero-sub {
        color: #5a655c !important;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }


    /* ======================================================
       NORMAL MAIN CONTENT
       ====================================================== */

    [data-testid="stAppViewContainer"] p {
        color: #1c2420;
    }


    /* ======================================================
       METRIC CARDS
       ====================================================== */

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
        color: #6b756e !important;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.15rem;
        font-weight: 500;
        color: #1c2420 !important;
        word-break: break-all;
    }


    /* ======================================================
       ANSWER BOX
       ====================================================== */

    .answer-box {
        background: #ffffff;
        border-left: 4px solid #2f6b4f;
        border-radius: 0 12px 12px 0;
        padding: 1.25rem 1.4rem;
        margin: 0.5rem 0 1.2rem 0;
        box-shadow: 0 1px 3px rgba(28, 36, 32, 0.06);
        line-height: 1.65;
        color: #1c2420 !important;
    }

    .answer-box p {
        color: #1c2420 !important;
    }


    /* ======================================================
       PAPER CHIP
       ====================================================== */

    .paper-chip {
        display: inline-block;
        background: #eef3ef;
        border: 1px solid #c9d4cb;
        color: #1c2420 !important;
        border-radius: 8px;
        padding: 0.45rem 0.75rem;
        margin: 0.25rem 0.35rem 0.25rem 0;
        font-size: 0.88rem;
    }


    /* ======================================================
       BUTTONS
       ====================================================== */

    div[data-testid="stButton"] > button {
        background: #2f6b4f;
        color: #ffffff !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
    }

    div[data-testid="stButton"] > button p,
    div[data-testid="stButton"] > button span {
        color: #ffffff !important;
    }

    div[data-testid="stButton"] > button:hover {
        background: #255640;
        color: #ffffff !important;
        border: none;
    }


    /* ======================================================
       FILE UPLOADER
       ====================================================== */

    [data-testid="stFileUploader"] section {
        background: #ffffff !important;
        border: 1px solid #d8ddd6 !important;
        border-radius: 10px !important;
    }

    [data-testid="stFileUploader"] section * {
        color: #1c2420 !important;
    }

    [data-testid="stFileUploader"] button {
        background: #eef3ef !important;
        color: #1c2420 !important;
        border: 1px solid #c9d4cb !important;
    }

    [data-testid="stFileUploader"] small {
        color: #647067 !important;
    }


    /* ======================================================
       SELECTBOX
       ====================================================== */

    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1px solid #d8ddd6 !important;
        color: #1c2420 !important;
    }

    div[data-baseweb="select"] input {
        color: #1c2420 !important;
    }

    div[data-baseweb="select"] span {
        color: #1c2420 !important;
    }

    div[data-baseweb="select"] svg {
        fill: #1c2420 !important;
    }


    /* ======================================================
       DROPDOWN MENU
       ====================================================== */

    ul[role="listbox"] {
        background: #ffffff !important;
    }

    ul[role="listbox"] li {
        color: #1c2420 !important;
        background: #ffffff !important;
    }

    ul[role="listbox"] li:hover {
        background: #eef3ef !important;
    }


    /* ======================================================
       TEXT AREA
       ====================================================== */

    textarea {
        background: #ffffff !important;
        color: #1c2420 !important;
        border: 1px solid #d8ddd6 !important;
        border-radius: 10px !important;
        caret-color: #1c2420 !important;
    }

    textarea::placeholder {
        color: #7a847d !important;
        opacity: 1 !important;
    }


    /* ======================================================
       TEXT INPUTS
       ====================================================== */

    input {
        color: #1c2420 !important;
        background: #ffffff !important;
    }

    input::placeholder {
        color: #7a847d !important;
        opacity: 1 !important;
    }


    /* ======================================================
       LABELS
       ====================================================== */

    [data-testid="stWidgetLabel"] p {
        color: #1c2420 !important;
        font-weight: 500;
    }


    /* ======================================================
       CAPTIONS
       ====================================================== */

    [data-testid="stCaptionContainer"] {
        color: #647067 !important;
    }

    [data-testid="stCaptionContainer"] p {
        color: #647067 !important;
    }


    /* ======================================================
       EXPANDERS
       ====================================================== */

    [data-testid="stExpander"] {
        border-color: #d8ddd6;
    }

    [data-testid="stExpander"] p,
    [data-testid="stExpander"] span {
        color: #1c2420;
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


def get_llm_provider():

    try:

        if "LLM_PROVIDER" in st.secrets:

            return str(
                st.secrets["LLM_PROVIDER"]
            ).lower()

    except Exception:
        pass

    return os.getenv(
        "LLM_PROVIDER",
        "ollama",
    ).lower()


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

    provider = get_llm_provider()

    if provider in (
        "groq",
        "openai",
    ):

        llm_description = (
            "Hybrid retrieval + reranking + hosted LLM"
        )

    else:

        llm_description = (
            "Hybrid retrieval + reranking + local LLM"
        )

    st.sidebar.caption(
        llm_description
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

    if provider == "groq":

        groq_model = "Groq hosted model"

        try:

            if "GROQ_MODEL" in st.secrets:

                groq_model = str(
                    st.secrets["GROQ_MODEL"]
                )

        except Exception:
            pass

        st.sidebar.text(
            f"LLM: {groq_model}"
        )

    elif provider == "openai":

        openai_model = "OpenAI hosted model"

        try:

            if "OPENAI_MODEL" in st.secrets:

                openai_model = str(
                    st.secrets["OPENAI_MODEL"]
                )

        except Exception:
            pass

        st.sidebar.text(
            f"LLM: {openai_model}"
        )

    else:

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

    safe_answer = html.escape(
        str(answer)
    )

    st.markdown(
        '<div class="answer-box">'
        + safe_answer.replace(
            "\n",
            "<br>"
        )
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

        safe_title = html.escape(
            str(title)
        )

        safe_preview = html.escape(
            str(preview)
        )

        safe_chunk_id = html.escape(
            str(chunk_id)
        )

        # ----------------------------------------------------
        # Use Streamlit's dedicated HTML renderer.
        # Do NOT use st.markdown() here.
        # ----------------------------------------------------

        source_html = f"""
        <div style="
            background: #ffffff;
            border: 1px solid #d8ddd6;
            border-radius: 12px;
            padding: 20px 22px;
            margin: 10px 0;
            box-shadow: 0 1px 3px rgba(28, 36, 32, 0.06);
            font-family: 'DM Sans', sans-serif;
        ">

            <div style="
                color: #1c2420;
                font-size: 20px;
                font-weight: 700;
                line-height: 1.4;
                margin-bottom: 8px;
            ">
                #{rank} — {safe_title}
            </div>

            <div style="
                color: #647067;
                font-size: 13px;
                line-height: 1.5;
                margin-bottom: 12px;
                font-family: 'IBM Plex Mono', monospace;
            ">
                Page {page}
                &nbsp;·&nbsp;
                Chunk {safe_chunk_id}
                &nbsp;·&nbsp;
                Score {score:.4f}
            </div>

            <div style="
                color: #3a433d;
                font-size: 15px;
                line-height: 1.65;
            ">
                {safe_preview}
            </div>

        </div>
        """

        st.html(
            source_html
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

        safe_name = html.escape(
            Path(name).stem
        )

        st.markdown(
            f"""
            <span class="paper-chip">
                {safe_name}
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

    generated_answer = html.escape(
        str(
            item.get(
                "generated_answer",
                "",
            )
        )
    )

    st.markdown(
        f'<div class="answer-box">'
        f'{generated_answer.replace(chr(10), "<br>")}'
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