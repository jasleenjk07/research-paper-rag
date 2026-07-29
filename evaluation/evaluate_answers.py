

import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.rag_pipeline import RAGPipeline


QUESTION_FILE = "evaluation/questions.json"
OUTPUT_FILE = "evaluation/generated_answers.json"


with open(QUESTION_FILE, "r") as file:
    questions = json.load(file)


pipeline = RAGPipeline()

records = []

for item in questions:
    question = item["question"]

    print(f"Generating answer for: {question}")

    answer = pipeline.answer(question)

    contexts = []
    if hasattr(pipeline, "last_retrieved_documents"):
        contexts = [
            doc.page_content
            for doc, _ in pipeline.last_retrieved_documents
        ]

    records.append(
        {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item.get("ground_truth", ""),
        }
    )


with open(OUTPUT_FILE, "w") as file:
    json.dump(records, file, indent=2)


ragas_dataset = Dataset.from_list(records)


results = evaluate(
    dataset=ragas_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
)


print("\n========== RAG Evaluation ==========")
print(results)