
# main.py

import os
from dotenv import load_dotenv
from agent import run_agent

# RAG imports
from rag import (
    ingest_document,
    get_indexed_documents,
    remove_document,
)

# Load variables from .env
load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")

if not LLM_API_KEY:
    raise ValueError("LLM_API_KEY is missing from .env")

if not LLM_MODEL:
    raise ValueError("LLM_MODEL is missing from .env")


def get_argument(question: str):
    """Return the text after the command, with surrounding quotes removed."""

    parts = question.split(maxsplit=1)

    if len(parts) != 2:
        return None

    value = parts[1].strip()

    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in "\"'"
    ):
        value = value[1:-1].strip()

    return value or None


def main():
    print("================================")
    print("    AI STUDY ASSISTANT")
    print("================================")
    print()

    print("Commands:")
    print('attach "file_path" - Add a document to the knowledge base')
    print("documents          - Show indexed documents")
    print('remove "filename"  - Remove a document')
    print("exit               - Exit application")
    print()

    message_history = []

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() == "exit":
            print("Goodbye!")
            break

        # ---------------------------------------------------
        # RAG: ATTACH DOCUMENT
        # ---------------------------------------------------

        if question.lower().startswith("attach "):
            try:
                file_path = get_argument(question)

                if not file_path:
                    print('Usage: attach "file_path"')
                    continue

                print("\nReading and indexing document...")

                result = ingest_document(file_path)

                print("\nRAG:")
                print(result["message"])
                print(f"Total characters: {result['total_characters']}")
                print(f"Total chunks: {result['total_chunks']}")
                print()

            except Exception as error:
                print("\nDocument attachment failed:")
                print(repr(error))
                print()

            continue

        # ---------------------------------------------------
        # RAG: SHOW ATTACHED DOCUMENTS
        # ---------------------------------------------------

        if question.lower() == "documents":

            documents = get_indexed_documents()

            print("\nIndexed documents:")

            if not documents:
                print("No documents indexed.")

            for document in documents:
                print(
                    f"- {document['filename']} "
                    f"({document['total_chunks']} chunks)"
                )

            print()
            continue

        # ---------------------------------------------------
        # RAG: REMOVE DOCUMENT
        # ---------------------------------------------------

        if question.lower().startswith("remove "):
            try:
                filename = get_argument(question)

                if not filename:
                    print('Usage: remove "filename"')
                    continue

                result = remove_document(filename)

                print("\nRAG:")
                print(result["message"])
                print()

            except Exception as error:
                print("\nDocument removal failed:")
                print(repr(error))
                print()

            continue

        # ---------------------------------------------------
        # ORIGINAL AI AGENT
        # ---------------------------------------------------

        try:
            print("Thinking...")

            output_data, message_history = run_agent(
                question,
                message_history,
            )

            print()
            print("AI:")
            print(f"Answer: {output_data.get('answer')}")
            print(f"Route: {output_data.get('route')}")
            print(f"Topic: {output_data.get('topic')}")
            print(f"Difficulty: {output_data.get('difficulty')}")
            print(f"Used calculator: {output_data.get('used_calculator')}")
            print(f"Used weather: {output_data.get('used_weather')}")
            print(f"Used web search: {output_data.get('used_web_search')}")
            print(f"Used knowledge: {output_data.get('used_knowledge')}")

            sources = output_data.get("sources") or []

            if sources:
                print(f"Sources: {', '.join(sources)}")

            print()

        except Exception as error:
            print()
            print("Something went wrong:")
            print(repr(error))
            print()


if __name__ == "__main__":
    main()
