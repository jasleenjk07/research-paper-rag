def get_paper_title(document):
    """
    Return the clean title stored in document metadata.

    Falls back to paper_name when a title is unavailable.
    """

    title = document.metadata.get("title")

    if title:
        return title.strip()

    return document.metadata.get(
        "paper_name",
        "Unknown Document"
    )


def build_context(retrieved_documents):
    """
    Build the context that will be provided to the LLM.

    Each retrieved chunk includes its document title,
    page number, and extracted content.
    """

    context_parts = []

    for document, score in retrieved_documents:

        title = get_paper_title(
            document
        )

        page = (
            document.metadata.get(
                "page",
                0
            ) + 1
        )

        context_part = (
            f"[{title}, Page {page}]\n\n"
            f"{document.page_content}"
        )

        context_parts.append(
            context_part
        )

    return "\n\n".join(
        context_parts
    )


def build_prompt(question, context):
    """
    Build a grounded prompt for the local LLM.
    """

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information
contained in the retrieved context below.

Instructions:

1. Read all retrieved passages carefully.
2. Use only information explicitly supported by the context.
3. Ignore passages that are unrelated to the question.
4. If the context contains relevant information, answer the
   question directly and naturally.
5. Do not use outside knowledge.
6. Do not invent facts, explanations, document titles,
   or page numbers.
7. When making an important factual claim, cite the supporting
   passage using this exact format:

   [Document Title, Page X]

8. Use only document titles and page numbers provided in
   the retrieved context.
9. Place citations immediately after the claim they support.
10. If multiple passages support the answer, combine them
    when useful.
11. Keep the answer concise and focused on the question.
12. Do not begin the response with phrases such as
    "Based on the provided context" or
    "According to the retrieved documents".
13. Do not add an "Answer:" heading.
14. If the retrieved context contains useful information
    that answers the question, answer it.
15. Only use the following fallback when the retrieved
    context contains no useful information for the question:

    The provided documents do not contain enough information
    to answer this question.

Retrieved Context:
------------------

{context}

------------------

Question:
{question}

Respond with only the answer.
"""

    return prompt


def generate_answer(
    question,
    retrieved_documents,
    llm
):
    """
    Generate a grounded answer from retrieved documents.
    """

    context = build_context(
        retrieved_documents
    )

    prompt = build_prompt(
        question,
        context
    )

    response = llm.invoke(
        prompt
    )

    answer = response.content.strip()

    # --------------------------------------------------------
    # Safety cleanup
    #
    # Qwen may occasionally add "Answer:" even though the
    # prompt tells it not to. Remove only that heading.
    # --------------------------------------------------------

    if answer.lower().startswith(
        "answer:"
    ):

        answer = answer[
            len("answer:"):
        ].strip()

    return answer