def build_context(retrieved_documents):
    context_parts = []

    for document, score in retrieved_documents:
        title = get_paper_title(document)

        page = document.metadata.get("page", 0) + 1

        context_part = f"""
[{title}, Page {page}]

{document.page_content}
"""

        context_parts.append(context_part)

    return "\n".join(context_parts)

def build_prompt(question, context):
    prompt = f"""
You are a research paper assistant.

Answer the user's question using only the provided context
from retrieved research papers.

Rules:
1. Use only information explicitly supported by the provided context.
2. Do not use external knowledge.
3. Cite each important claim using the paper title and page number.
4. Use the exact citation format:
   [Paper Title, Page X]
5. Use only paper titles and page numbers provided in the context.
6. Do not invent citations or page numbers.
7. Place the citation immediately after the claim it supports.
8. If the context does not contain enough information to answer
   the question, say exactly:
   "The provided research papers do not contain enough information
   to answer this question."

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt

def generate_answer(question, retrieved_documents, llm):
    context = build_context(retrieved_documents)
    prompt = build_prompt(question, context)
    response = llm.invoke(prompt)
    return response.content

def get_paper_title(document):
    title = document.metadata.get("title")
    if title:
        return title.strip()
    return document.metadata.get(
        "paper_name",
        "Unknown Paper"
    )