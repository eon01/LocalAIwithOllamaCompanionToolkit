# smoke.py
"""Confirms the SDK can talk to the local Ollama server.

Run with: uv run smoke.py
"""

# importlib.metadata reads version info from installed package metadata.
# We use this because the `ollama` package does not expose a __version__
# attribute. version("ollama") returns the same string as `pip show ollama`.
from importlib.metadata import version

# Client is the sync HTTP client. ResponseError is what the SDK raises when
# the server returns an HTTP error (404 model not found, 500 server error, etc).
from ollama import Client, ResponseError

# OLLAMA_HOST comes from .env via config.py. Centralizing it means we change
# the host in one place when switching between local and a remote server.
from config import OLLAMA_HOST


def main() -> None:
    # One Client per process. It holds an httpx connection pool internally,
    # so reusing the same instance across requests is meaningfully faster
    # than constructing a new Client for every call.
    client = Client(host=OLLAMA_HOST)

    try:
        # client.list() hits GET /api/tags and returns a ListResponse.
        # ListResponse.models is a list of Model objects, each with .model
        # (the tag), .size (bytes), .modified_at (datetime), and .details.
        resp = client.list()
    except ResponseError as e:
        # The server is reachable but returned an HTTP error.
        # e.status_code is the HTTP code, e.error is the message string.
        print(f"Server error {e.status_code}: {e.error}")
        raise SystemExit(1)
    except ConnectionError:
        # Could not reach the server at all. Usually means Ollama is not
        # running, is bound to a different host/port, or a firewall is
        # blocking the connection.
        print(f"Could not connect to Ollama at {OLLAMA_HOST}")
        print("Check: systemctl is-active ollama")
        raise SystemExit(1)

    # Three sanity checks in one place: the SDK version we have installed,
    # the host we are talking to, and proof we got real data back.
    print(f"SDK version:    {version('ollama')}")
    print(f"Server host:    {OLLAMA_HOST}")
    print(f"Models on disk: {len(resp.models)}")


if __name__ == "__main__":
    # The standard "run as a script" guard. Lets this file be imported
    # without main() firing, which we will want when the REPL lessons
    # reuse helpers from here.
    main()
