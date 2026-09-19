"""Answer generation for the RAG pipeline."""

from langchain_openai import ChatOpenAI
from langchain.schema import Document

from utils.config import get_settings


def _build_context(documents: list[Document]) -> str:
    """Merge retrieved chunks into a single prompt-ready context string."""
    context_blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        context_blocks.append(f"[Source {index}: {source}]\n{document.page_content}")
    return "\n\n".join(context_blocks)


def _build_history(chat_history: list[dict]) -> str:
    """Format chat history so the generator can keep conversational context."""
    if not chat_history:
        return "No previous conversation."
    return "\n".join(f"{item['role'].title()}: {item['content']}" for item in chat_history[-6:])


def _extractive_answer(question: str, documents: list[Document]) -> str:
    """Create a simple answer without an external LLM for low-resource fallback."""
    if not documents:
        return "I could not find relevant information in the uploaded documents."

    best_passage = documents[0].page_content.strip()
    preview = best_passage[:900]
    return (
        "Low-resource answer mode is enabled, so I am answering directly from the best "
        f"retrieved passage.\n\nQuestion: {question}\n\nRelevant context:\n{preview}"
    )


def generate_answer(
    question: str,
    documents: list[Document],
    chat_history: list[dict],
    model_name: str = "openai",
) -> tuple[str, list[dict]]:
    """Generate the final answer and return source metadata for the frontend."""
    sources = [
        {
            "source": document.metadata.get("source", "unknown"),
            "chunk_id": document.metadata.get("chunk_id"),
            "preview": document.page_content[:180],
        }
        for document in documents
    ]

    if model_name == "extractive":
        return _extractive_answer(question=question, documents=documents), sources

    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is not configured. Use the 'extractive' model or add your API key."
        )

    if not documents:
        return (
            "I could not find a confident enough match in the indexed documents. "
            "Try rephrasing the question or upload a more relevant document."
        ), sources

    prompt = f"""
You are an AI Knowledge Assistant that answers questions using only the provided document context.
If the answer is not present in the context, clearly say that the documents do not contain the answer.

Conversation history:
{_build_history(chat_history)}

Context:
{_build_context(documents)}

Question:
{question}

Write a concise, beginner-friendly answer and mention when information is inferred.
""".strip()

    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=0.2,
    )
    response = llm.invoke(prompt)
    return response.content, sources
