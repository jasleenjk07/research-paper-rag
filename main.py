from src.rag_pipeline import RAGPipeline

def main():
    rag = RAGPipeline()
    question = "How does LoRA reduce trainable parameters?"
    print("\nQuestion:")
    print(question)
    answer = rag.answer(question)
    print("\nAnswer:")
    print(answer)

if __name__ == "__main__":
    main()