# mycli.py
"""Small CLI that mirrors `ollama list/show/pull/ps` using the Python SDK.

Run it with: uv run mycli <subcommand> [args]
"""

import argparse
from datetime import datetime, timezone

from ollama import Client

from config import OLLAMA_HOST

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def human_size(n_bytes: int) -> str:
    """Bytes to a short human string. SDK gives sizes in raw bytes."""
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def human_age(when: datetime) -> str:
    """A datetime to a rough 'N units ago' string. Good enough for a CLI."""
    # The SDK gives timezone-aware datetimes. Compare in UTC to avoid drift.
    now = datetime.now(timezone.utc)
    delta = now - when
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def print_table(rows: list[list[str]], headers: list[str]) -> None:
    """Print a simple aligned table. Avoids pulling in `rich` or `tabulate`."""
    if not rows:
        print("(no rows)")
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    for row in rows:
        print(fmt.format(*row))


# -----------------------------------------------------------------------------
# Subcommands
# -----------------------------------------------------------------------------


def cmd_list(client: Client, _args: argparse.Namespace) -> int:
    """List models on disk. Same data as `ollama list` and GET /api/tags."""
    resp = client.list()
    rows = [
        [
            m.model,  # the tag, e.g. "granite3.3:2b"
            human_size(m.size),  # bytes -> "1.5 GB"
            human_age(m.modified_at),  # datetime -> "2h ago"
        ]
        for m in resp.models
    ]
    print_table(rows, ["NAME", "SIZE", "MODIFIED"])
    return 0


def cmd_show(client: Client, args: argparse.Namespace) -> int:
    """Print the metadata for one model. Same data as `ollama show`."""
    resp = client.show(args.model)

    # `details` is a small object with architecture/quantization/etc.
    # `model_info` is the full raw metadata dict from the GGUF; we pull the
    # context length out of it. The key is prefixed by architecture, e.g.
    # "granite.context_length" for granite, "llama.context_length" for llama.
    arch = resp.details.family
    ctx_key = f"{arch}.context_length"
    context = resp.modelinfo.get(ctx_key) if resp.modelinfo else None

    print(f"Architecture:  {arch}")
    print(f"Parameters:    {resp.details.parameter_size}")
    print(f"Quantization:  {resp.details.quantization_level}")
    print(f"Context:       {context if context is not None else 'unknown'}")
    if resp.capabilities:
        print(f"Capabilities:  {', '.join(resp.capabilities)}")
    return 0


def cmd_pull(client: Client, args: argparse.Namespace) -> int:
    """Pull a model, printing each progress event."""
    for event in client.pull(args.model, stream=True):
        if event.total and event.completed is not None:
            print(f"{event.status}: {event.completed}/{event.total}")
        else:
            print(event.status)
    return 0


def cmd_ps(client: Client, _args: argparse.Namespace) -> None:
    """List models currently loaded in memory."""
    print(f"{'NAME':<20} {'SIZE':<10} {'GPU':<6} EXPIRES")
    for m in client.ps().models:
        gpu_pct = (m.size_vram / m.size * 100) if m.size else 0
        print(
            f"{m.model:<20} {human_size(m.size):<10} {gpu_pct:>4.0f}% {m.expires_at.astimezone()}"
        )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(prog="mycli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")
    sub.add_parser("ps")
    sub.add_parser("show").add_argument("model")
    sub.add_parser("pull").add_argument("model")

    args = parser.parse_args()
    client = Client(host=OLLAMA_HOST)

    handlers = {"list": cmd_list, "show": cmd_show, "pull": cmd_pull, "ps": cmd_ps}
    handlers[args.cmd](client, args)


if __name__ == "__main__":
    raise SystemExit(main())
