# repl.py (pass 4: streaming + multi-line input + history trimming)
"""A chat REPL with streaming output, multi-line input, and a sliding
window over the conversation history.

Multi-line input: type your message across as many lines as you like,
then send it by hitting Enter on an empty line. Type /bye on its own
line to exit.

History trimming: once the total content across all messages exceeds
MAX_HISTORY_CHARS, drop the oldest user/assistant pairs until we're
back under budget. A leading system message, if present, is never
dropped.
"""

from ollama import Client

from config import OLLAMA_HOST, OLLAMA_MODEL

# Maximum total characters across all messages. Roughly four characters
# per token for English, so 8000 characters is around 2000 tokens. Adjust
# based on your model's context window and how much room you want to
# leave for the reply.
MAX_HISTORY_CHARS = 8000


def trim_history(
    messages: list[dict], max_chars: int = MAX_HISTORY_CHARS
) -> list[dict]:
    """Drop oldest user/assistant pairs until total content fits in max_chars.

    A system message at index 0, if present, is preserved no matter what.
    We drop in pairs (user + assistant) because dropping only one half of
    a turn leaves the model with an orphan reply or an unanswered question,
    which confuses it more than dropping both.
    """
    # Preserve a leading system message if there is one.
    has_system = bool(messages) and messages[0].get("role") == "system"
    head = messages[:1] if has_system else []
    body = messages[1:] if has_system else messages[:]

    def total_chars(msgs: list[dict]) -> int:
        return sum(len(m.get("content", "")) for m in msgs)

    while total_chars(head + body) > max_chars and len(body) >= 2:
        # Drop the oldest pair (one user, one assistant).
        body = body[2:]

    return head + body


def read_input() -> str:
    """Read lines from stdin until the user submits an empty line.

    The first line uses a '> ' prompt, continuation lines use '  ' so
    the user can see they're still inside the same message. Returns the
    joined message with newlines preserved.
    """
    lines: list[str] = []
    prompt = "> "
    while True:
        line = input(prompt)
        # Empty line means "I'm done, send it". This applies even on the
        # first line: hitting Enter immediately just gives an empty turn,
        # which the caller can ignore.
        if line == "":
            break
        lines.append(line)
        prompt = "  "
    return "\n".join(lines)


def main() -> None:
    # We bump the client timeout because long replies can easily exceed
    # the default. Setting it to None disables the timeout entirely; we
    # use a generous number instead so a truly stuck server still fails.
    client = Client(host=OLLAMA_HOST, timeout=300)

    print(f"Chatting with {OLLAMA_MODEL}.")
    print("Hit Enter on an empty line to send. Type /bye to exit.")

    messages: list[dict] = []

    while True:
        user = read_input()

        # Skip empty submissions instead of sending an empty turn to the
        # model, which wastes a round trip and confuses some models.
        if user == "":
            continue
        if user.strip() == "/bye":
            break

        messages.append({"role": "user", "content": user})

        # Apply the sliding window before sending. Doing this here, after
        # appending the user's turn but before the model call, ensures
        # the most recent user message is always included even if it
        # alone would push us over budget.
        messages = trim_history(messages)

        # stream=True returns an iterator of ChatResponse chunks. Each
        # chunk has a small piece of text in chunk.message.content. We
        # print it immediately and also accumulate it so we have the
        # complete reply to append to history when the stream ends.
        full_reply = ""
        for chunk in client.chat(model=OLLAMA_MODEL, messages=messages, stream=True):
            piece = chunk.message.content
            print(piece, end="", flush=True)
            full_reply += piece

        # Print a final newline so the next "> " prompt starts on its
        # own line. Without this the prompt would butt up against the
        # last character of the reply.
        print()

        messages.append({"role": "assistant", "content": full_reply})


if __name__ == "__main__":
    main()

