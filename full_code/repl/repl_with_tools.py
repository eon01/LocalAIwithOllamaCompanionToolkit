# repl.py (pass 7: long-term memory + tool calling)
"""
A chat REPL that remembers facts about the user across sessions
and can call tools to fetch fresh data from the open web.
"""

import os
import threading

import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
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


# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------
#
# Each function decorated with @tool becomes something the agent can call.
# The docstring is what the model reads to decide whether to call the tool.
# Keep them short, specific, and action-oriented; the model picks tools by
# matching the user's intent against these descriptions.
#
# Tool return values become tool-message content in the conversation, which
# the model then reads and turns into a natural-language reply.


def _get_coordinates(location: str) -> tuple[float, float]:
    """Resolve a place name to (latitude, longitude) via Open-Meteo geocoding.

    Underscore prefix marks this as a helper that's NOT exposed to the model.
    The two public tools below call it before hitting their respective APIs.
    """
    response = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    data = response.json()
    if "results" in data and data["results"]:
        first = data["results"][0]
        return first["latitude"], first["longitude"]
    raise ValueError(f"Could not find coordinates for location: {location}")


@tool
def get_air_quality(location: str) -> str:
    """Get current air quality (PM10 and PM2.5) for a named location."""
    latitude, longitude = _get_coordinates(location)
    response = httpx.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "pm10,pm2_5",
            "forecast_days": 1,
        },
        timeout=10,
    )
    data = response.json()
    if "hourly" in data and "pm10" in data["hourly"] and "pm2_5" in data["hourly"]:
        pm10 = data["hourly"]["pm10"][0]  # index 0 = current hour
        pm2_5 = data["hourly"]["pm2_5"][0]
        result = f"PM10: {pm10} μg/m³, PM2.5: {pm2_5} μg/m³"
    else:
        result = "Air quality data not available"
    return f"Air quality in {location}: {result}"


@tool
def get_temperature(location: str) -> str:
    """Get the current temperature in Celsius for a named location."""
    latitude, longitude = _get_coordinates(location)
    response = httpx.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m",
            "forecast_days": 1,
        },
        timeout=10,
    )
    data = response.json()
    if "hourly" in data and "temperature_2m" in data["hourly"]:
        temperature = data["hourly"]["temperature_2m"][0]  # index 0 = current hour
        result = f"Temperature: {temperature} °C"
    else:
        result = "Temperature data not available"
    return f"Temperature in {location}: {result}"


# The list passed to create_agent. Add new tools here and the agent
# will see them automatically.
TOOLS = [get_air_quality, get_temperature]


# -----------------------------------------------------------------------------
# Memory layer (unchanged from pass 6)
# -----------------------------------------------------------------------------


def build_memory() -> Memory:
    """Build a fully local mem0 instance backed by Ollama and Chroma."""
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
    """Search memory and return the top-k facts as a formatted string."""
    results = memory.search(query=query, filters={"user_id": user_id}, limit=k)
    items = results.get("results", results)
    if not items:
        return ""
    lines = [f"- {m['memory']}" for m in items]
    return "Known facts about the user:\n" + "\n".join(lines)


def write_memory_async(memory: Memory, user_text: str, reply: str) -> None:
    """Fire mem0.add() in a background thread, tracked for clean shutdown."""

    def _run() -> None:
        try:
            print("\nMemory write started...", flush=True)
            memory.add(
                messages=[
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": reply},
                ],
                user_id=USER_ID,
                infer=False,
            )
            print("\nMemory write complete.", flush=True)
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


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------


def main() -> None:
    global USER_ID
    USER_ID = input("Enter your user id: ").strip() or "default"

    # The worker model handles each turn of the chat AND the tool-call
    # decisions. Granite 3.3 supports tool calling; check with
    # `ollama show <model>` and look for "tools" in capabilities.
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_HOST,
        num_predict=512,
    )

    summarizer = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_HOST,
        num_predict=512,
    )

    # create_agent now takes tools=TOOLS. The agent's internal graph
    # routes a single turn through a tool-call loop: model decides if
    # any tool should fire, runs it, feeds the result back, decides
    # again, eventually produces a final assistant reply.
    agent = create_agent(
        model=llm,
        tools=TOOLS,
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

    print(f"Chatting with {OLLAMA_MODEL} (with long-term memory and tools).")
    print(f"Tools available: {', '.join(t.name for t in TOOLS)}")
    print("Hit Enter on an empty line to send. Type /bye to exit.")
    print("Waiting for pending memory writes is automatic on /bye.\n")

    while True:
        user = read_input()
        if user == "":
            continue

        if user.strip() == "/bye":
            pending = [t for t in _write_threads if t.is_alive()]
            if pending:
                print(f"Waiting for {len(pending)} memory write(s) to finish...")
                for t in pending:
                    t.join()
            break

        memory_block = relevant_memories(memory, query=user, user_id=USER_ID)

        messages: list = []
        if memory_block:
            messages.append(SystemMessage(content=memory_block))
        messages.append(HumanMessage(content=user))

        # The agent loop is more complex now. With tools wired in, a
        # single user turn can produce multiple events:
        #   - AIMessage with tool_calls but empty content (model decided
        #     to call a tool)
        #   - ToolMessage (tool result)
        #   - AIMessage with content (final natural-language reply)
        # We stream only the visible reply text. Tool calls happen
        # silently in the background; the user just sees the final
        # answer that incorporates tool results.
        full_reply = ""
        for chunk, _ in agent.stream(
            {"messages": messages},
            stream_mode="messages",
        ):
            # Skip tool-call planning chunks (they have tool_calls but
            # empty content) and tool result chunks. Show only the
            # streaming text of the final assistant message.
            if isinstance(chunk, AIMessage) and chunk.content:
                print(chunk.content, end="", flush=True)
                full_reply += chunk.content
        print()

        # Only write the user message and the final visible reply to
        # memory. The tool-call/result roundtrip is implementation
        # detail and doesn't belong in the user's long-term memory.
        write_memory_async(memory, user, full_reply)


if __name__ == "__main__":
    main()

