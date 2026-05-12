# repl.py (pass 6: long-term memory with mem0)
"""
A chat REPL that remembers facts about the user across sessions.
"""

import os
import threading

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from mem0 import Memory

from config import OLLAMA_HOST, OLLAMA_MODEL

# Who the memories belong to.
USER_ID = input("Enter your user id: ")

# Where mem0 keeps its vector store on disk. Survives restarts.
MEMORY_DB_PATH = os.path.expanduser("~/.ollama-python/mem0_db")

# Models used for fact extraction
EXTRACTION_MODEL = "granite3.3:2b"
# Model used for text embedding
EMBED_MODEL = "nomic-embed-text:v1.5"
# Collection name in Chroma (mem0's vector store)
COLLECTION_NAME = "repl_memories"


def build_memory() -> Memory:
    """Build a fully local mem0 instance backed by Ollama and Chroma.

    Three components are configured:
      - llm: extracts facts from conversations.
      - embedder: turns text into vectors to perform vector search.
      - vector_store: where the vectors live. Chroma is the simplest
        option.
    """
    config = {
        "llm": {
            "provider": "ollama",
            "config": {
                "model": EXTRACTION_MODEL,
                "ollama_base_url": OLLAMA_HOST,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": EMBED_MODEL,
                "ollama_base_url": OLLAMA_HOST,
            },
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": COLLECTION_NAME,
                "path": MEMORY_DB_PATH,
            },
        },
    }
    return Memory.from_config(config)


def relevant_memories(memory: Memory, query: str, user_id: str, k: int = 5) -> str:
    """Search memory and format the top-k results as a string for the prompt.

    Returns an empty string if nothing relevant was found, so the caller
    can skip the system message entirely in that case.
    """
    # search for facts relevant to the user's question
    results = memory.search(query=query, filters={"user_id": user_id}, limit=k)

    # Get the results as a list of dicts.
    items = results.get("results", results)  # if isinstance(results, dict) else results
    if not items:
        return ""

    lines = [f"- {m['memory']}" for m in items]
    return "Known facts about the user:\n" + "\n".join(lines)


def write_memory_async(memory: Memory, user_text: str, reply: str) -> None:
    """Fire mem0.add() in a background thread.

    Fact extraction calls the configured LLM internally.
    """

    def _run() -> None:
        try:
            # Write to the memory store
            memory.add(
                messages=[
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": reply},
                ],
                user_id=USER_ID,
            )
        except Exception as e:
            print(f"\n(memory write failed: {e})")

    # daemon=True so the thread dies with the process; /bye exits cleanly
    # even if a write is mid-flight.
    threading.Thread(target=_run, daemon=True).start()


def read_input() -> str:
    """Read lines from stdin until the user submits an empty line."""
    lines: list[str] = []
    prompt = "> "
    while True:
        line = input(prompt)
        if line == "":
            break
        lines.append(line)
        prompt = "  "
    return "\n".join(lines)


def main() -> None:
    # The worker model handles each turn of the chat.
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_HOST,
        num_predict=512,
    )

    # The summarizer compresses old turns within the current session.
    summarizer = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_HOST,
        num_predict=512,
    )

    # The agent: chat model + in-session summarization middleware.
    agent = create_agent(
        model=llm,
        tools=[],
        middleware=[
            SummarizationMiddleware(
                model=summarizer,
                trigger=("tokens", 2000),
                keep=("messages", 6),
            ),
        ],
    )

    print("Loading memory...")
    # Load the memory store
    memory = build_memory()

    print(f"Chatting with {OLLAMA_MODEL} (with long-term memory).")
    print("Hit Enter on an empty line to send. Type /bye to exit.")

    while True:
        user = read_input()
        if user == "":
            continue
        if user.strip() == "/bye":
            break

        print("[searching memory...]", flush=True)
        memory_block = relevant_memories(memory, query=user, user_id=USER_ID)

        messages: list = []
        # If there's a relevant memory block, inject it as a system message.
        if memory_block:
            messages.append(SystemMessage(content=memory_block))
        messages.append(HumanMessage(content=user))

        print("[calling agent.stream...]", flush=True)
        full_reply = ""
        for chunk, _ in agent.stream(
            {"messages": messages},
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessage) and chunk.content:
                print(chunk.content, end="", flush=True)
                full_reply += chunk.content
        print()
        print("[stream done]", flush=True)

        write_memory_async(memory, user, full_reply)


if __name__ == "__main__":
    main()

