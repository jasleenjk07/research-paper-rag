import json
import time
from pathlib import Path

from src.rag_pipeline import RAGPipeline

QUESTION_FILE = "evaluation/questions.json"
RESULT_FILE = "evaluation/results.json"


def main():
    print("=" * 60)
    print("Loading RAG Pipeline...")
    print("=" * 60)

    pipeline = RAGPipeline()

    with open(QUESTION_FILE, "r") as f:
        questions = json.load(f)

    results = []

    retrieval_times = []
    generation_times = []
    total_times = []

    total_answer_words = 0
    total_contexts = 0

    print("\nStarting Evaluation...\n")

    for idx, item in enumerate(questions, start=1):

        question = item["question"]

        print(f"[{idx}/{len(questions)}] {question}")

        total_start = time.perf_counter()

        # -------------------------------
        # Run complete pipeline
        # -------------------------------
        answer = pipeline.answer(question)

        total_end = time.perf_counter()

        total_time = total_end - total_start

        # -------------------------------
        # Retrieved contexts
        # -------------------------------
        retrieved_docs = pipeline.last_retrieved_documents

        contexts = []
        citations = []

        for doc, score in retrieved_docs:

            contexts.append(doc.page_content)

            citations.append({
                "paper": doc.metadata.get("title"),
                "page": doc.metadata.get("page"),
                "score": float(score)
            })

        # ------------------------------------------------
        # Timing
        # ------------------------------------------------
        #
        # Since current pipeline.answer() doesn't separately
        # expose retrieval & generation timings,
        # we approximate for now.
        #
        retrieval_time = None
        generation_time = None

        retrieval_times.append(0)
        generation_times.append(0)
        total_times.append(total_time)

        total_answer_words += len(answer.split())
        total_contexts += len(contexts)

        results.append({

            "question": question,

            "generated_answer": answer,

            "ground_truth": item["ground_truth"],

            "expected_paper": item["expected_paper"],

            "category": item["category"],

            "difficulty": item["difficulty"],

            "retrieved_contexts": contexts,

            "citations": citations,

            "retrieval_time": retrieval_time,

            "generation_time": generation_time,

            "total_time": round(total_time, 3)

        })

        print(f"✓ Completed ({total_time:.2f} sec)\n")

    # -----------------------------------------
    # Save Results
    # -----------------------------------------

    Path("evaluation").mkdir(exist_ok=True)

    with open(RESULT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    # -----------------------------------------
    # Summary
    # -----------------------------------------

    avg_total = sum(total_times) / len(total_times)

    avg_answer_length = total_answer_words / len(results)

    avg_contexts = total_contexts / len(results)

    print("=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    print(f"Questions Evaluated     : {len(results)}")
    print(f"Average Total Time      : {avg_total:.2f} sec")
    print(f"Average Answer Length   : {avg_answer_length:.2f} words")
    print(f"Average Contexts Used   : {avg_contexts:.2f}")

    print(f"\nResults saved to {RESULT_FILE}")

    print("=" * 60)


if __name__ == "__main__":
    main()