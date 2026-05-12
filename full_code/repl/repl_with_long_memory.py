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

# Where mem0 keeps its vector store on disk. Survives restarts.
# Delete this directory to wipe all memories and start fresh.
MEMORY_DB_PATH = os.path.expanduser("~/.ollama-python/mem0_db")

# Model used by mem0 to extract durable facts from conversations.
# Needs to be smart enough to distinguish "the user said X" from
# generic assistant text. A 2B model is too small for reliable
# extraction; 3B is the practical minimum.
EXTRACTION_MODEL = "granite3.3:2b"

# Model used to turn text into vectors for similarity search.
EMBED_MODEL = "nomic-embed-text:v1.5"

# Collection name inside Chroma. One collection per app is fine.
COLLECTION_NAME = "repl_memories"

# Set inside main() once the user has typed their id at the prompt.
# Kept at module scope because the background-thread closure reads it.
USER_ID: str = ""

# Tracks every memory write thread we spawn. /bye joins these before
# exiting so writes finish landing on disk instead of being killed
# mid-call as orphan daemons.
_write_threads: list[threading.Thread] = []


def build_memory() -> Memory:
    """Build a fully local mem0 instance backed by Ollama and Chroma.

    Three components are configured:
      - llm: extracts facts from conversations (EXTRACTION_MODEL).
      - embedder: turns text into vectors for similarity search.
      - vector_store: where the vectors live. Chroma persists to disk
        with no extra services to run.
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
    """Search memory and return the top-k facts as a formatted string.

    Returns an empty string if nothing relevant was found, so the caller
    can skip the system message injection entirely in that case.
    """
    # mem0 v2: scoping moved from top-level kwargs into a filters dict.
    results = memory.search(query=query, filters={"user_id": user_id}, limit=k)

    # search() returns {"results": [...]} in v2; older versions returned
    # a bare list. Handle both.
    items = results.get("results", results)
    if not items:
        return ""

    lines = [f"- {m['memory']}" for m in items]
    return "Known facts about the user:\n" + "\n".join(lines)


def write_memory_async(memory: Memory, user_text: str, reply: str) -> None:
    """Fire mem0.add() in a background thread, tracked for clean shutdown.

    Use infer=False to skip mem0's LLM-based extraction and dedup
    steps. They are slow (two LLM calls) and unreliable on small local
    models (silently drop facts). Verbatim storage is fast, reliable,
    and good enough for vector search to find the right context later.
    """

    def _run() -> None:
        try:
            print("\nMemory write started...", flush=True)
            memory.add(
                messages=[
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": reply},
                ],
                user_id=USER_ID,
                infer=True,
            )
            print("\nMemory write complete...", flush=True)

        except Exception as e:
            print(f"\n(memory write failed: {e})", flush=True)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    _write_threads.append(t)


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
    # Prompt for user id inside main() instead of at module scope, so
    # importing repl.py from a debug script doesn't trigger an input()
    # prompt. Defaults to "default" if the user just hits Enter.
    global USER_ID
    USER_ID = input("Enter your user id: ").strip() or "default"

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

    # The agent wraps the chat model with in-session summarization
    # middleware. Long-term memory (mem0) handles cross-session facts;
    # the summarizer handles overflow within one running conversation.
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
    memory = build_memory()

    print(f"Chatting with {OLLAMA_MODEL} (with long-term memory).")
    print("Hit Enter on an empty line to send. Type /bye to exit.")
    print("Waiting for pending memory writes is automatic on /bye.\n")

    while True:
        user = read_input()
        if user == "":
            continue

        if user.strip() == "/bye":
            # Wait for any in-flight memory writes before exiting.
            # Without this, daemon threads get killed mid-call and the
            # facts from the most recent turn never land in Chroma.
            pending = [t for t in _write_threads if t.is_alive()]
            if pending:
                print(f"Waiting for {len(pending)} memory write(s) to finish...")
                for t in pending:
                    t.join()
            break

        # Search memory for facts relevant to this turn's question.
        memory_block = relevant_memories(memory, query=user, user_id=USER_ID)

        # Build the prompt for this turn: a system message containing
        # any relevant facts (if we have any), followed by the user's
        # message. We rebuild this list every turn because mem0 is now
        # the source of long-term memory; there's no in-Python history.
        messages: list = []
        if memory_block:
            messages.append(SystemMessage(content=memory_block))
        messages.append(HumanMessage(content=user))

        # Stream the reply, accumulating it for the memory write.
        full_reply = ""
        for chunk, _ in agent.stream(
            {"messages": messages},
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessage) and chunk.content:
                print(chunk.content, end="", flush=True)
                full_reply += chunk.content
        print()

        # Fire the memory write in the background. The next prompt
        # appears immediately; the write finishes asynchronously and
        # prints "(memory write done: ...)" when it lands.
        write_memory_async(memory, user, full_reply)


if __name__ == "__main__":
    main()

