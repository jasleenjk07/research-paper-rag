from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def _clean_title(title):
    """
    Clean and validate a possible PDF title.

    Returns None when the value is clearly a filename,
    identifier, or generic metadata value.
    """

    if not title:
        return None

    title = str(title).strip()

    if not title:
        return None

    title = " ".join(title.split())

    lower_title = title.lower()

    # Ignore filename-like values
    if lower_title.endswith(".pdf"):
        return None

    # Ignore generic metadata
    generic_titles = {
        "untitled",
        "document",
        "unknown",
        "unknown document",
    }

    if lower_title in generic_titles:
        return None

    # Ignore patent/document identifiers such as:
    # US00000010242255B220190326
    if (
        lower_title.startswith("us")
        and any(char.isdigit() for char in lower_title)
        and len(lower_title) > 10
    ):
        return None

    return title


def _extract_patent_title(pages):
    """
    Extract a patent title from the '(54)' patent field.
    """

    if not pages:
        return None

    text_parts = []

    for page in pages[:3]:
        text = page.page_content.strip()

        if text:
            text_parts.append(text)

    if not text_parts:
        return None

    text = " ".join(text_parts)

    # Normalize whitespace.
    text = " ".join(text.split())

    # Find the patent title marker.
    marker = "( 54 )"

    marker_position = text.find(marker)

    if marker_position == -1:
        # Some PDFs may extract it without spaces.
        marker = "(54)"
        marker_position = text.find(marker)

    if marker_position == -1:
        return None

    # Everything immediately after (54) starts the title.
    remaining = text[
        marker_position + len(marker):
    ].strip()

    # Patent bibliographic fields that indicate the title
    # has ended.
    stop_markers = [
        "( 8 )",
        "( 10 )",
        "( 45 )",
        "( 51 )",
        "( 52 )",
        "( 56 )",
        "( 57 )",
        "( 58 )",
        "( 63 )",
        "( 65 )",
        "( 73 )",
        "( 74 )",
        "( 75 )",
    ]

    # Find the earliest stop marker.
    stop_position = None

    for stop_marker in stop_markers:

        position = remaining.find(
            stop_marker
        )

        if position != -1:

            if (
                stop_position is None
                or position < stop_position
            ):
                stop_position = position

    if stop_position is not None:
        title = remaining[
            :stop_position
        ].strip()
    else:
        title = remaining[:200].strip()

    title = _clean_title(title)

    if not title:
        return None

    # The title should not be excessively long.
    if len(title) > 200:
        return None

    return title


def _extract_research_paper_title(pages):
    """
    Conservative title extraction for normal research papers.

    Used only when valid PDF metadata is unavailable.
    """

    if not pages:
        return None

    text_parts = []

    for page in pages[:2]:

        text = page.page_content.strip()

        if text:
            text_parts.append(text)

    if not text_parts:
        return None

    text = "\n".join(text_parts)

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return None

    for line in lines[:30]:

        cleaned = _clean_title(line)

        if not cleaned:
            continue

        lower = cleaned.lower()

        skip_prefixes = (
            "abstract",
            "introduction",
            "contents",
            "references",
            "copyright",
            "arxiv",
            "doi:",
            "http://",
            "https://",
            "united states patent",
            "us ",
        )

        if lower.startswith(skip_prefixes):
            continue

        if len(cleaned) < 10:
            continue

        if len(cleaned) > 200:
            continue

        if cleaned.endswith(
            (".", ":", ";")
        ):
            continue

        return cleaned

    return None


def load_pdf(pdf_path):
    """
    Load a PDF and attach consistent document metadata.

    Metadata includes:

        paper_name:
            Original filename stem.

        title:
            Clean document title.
    """

    pdf_path = Path(pdf_path)

    loader = PyPDFLoader(
        str(pdf_path)
    )

    pages = loader.load()

    paper_name = pdf_path.stem

    # ---------------------------------------------------------
    # Check whether this is a patent.
    # ---------------------------------------------------------

    first_pages_text = " ".join(
        page.page_content
        for page in pages[:3]
    )

    is_patent = (
        "( 54 )" in first_pages_text
        or "(54)" in first_pages_text
        or "United States Patent"
        in first_pages_text
    )

    # ---------------------------------------------------------
    # Try PDF metadata only for non-patents.
    #
    # Patent metadata frequently contains identifiers
    # instead of the actual invention title.
    # ---------------------------------------------------------

    metadata_title = None

    if pages and not is_patent:

        metadata_title = _clean_title(
            pages[0].metadata.get(
                "title"
            )
        )

    # ---------------------------------------------------------
    # Determine document title.
    # ---------------------------------------------------------

    if is_patent:

        document_title = (
            _extract_patent_title(
                pages
            )
            or paper_name
        )

    else:

        document_title = (
            metadata_title
            or _extract_research_paper_title(
                pages
            )
            or paper_name
        )

    # ---------------------------------------------------------
    # Attach consistent metadata to every page.
    # ---------------------------------------------------------

    for page in pages:

        page.metadata[
            "paper_name"
        ] = paper_name

        page.metadata[
            "title"
        ] = document_title

    return pages


def load_documents(folder_path):
    """
    Load all PDF documents from a folder.
    """

    documents = []

    folder = Path(
        folder_path
    )

    for pdf_path in sorted(
        folder.glob("*.pdf")
    ):

        pages = load_pdf(
            pdf_path
        )

        documents.extend(
            pages
        )

    return documents