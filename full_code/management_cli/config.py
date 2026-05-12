# config.py
"""Central config. Loads .env once, exposes constants.

Real environment variables win over .env values, so a shell `export
OLLAMA_HOST=...` overrides whatever is in the file. That's what you
want when switching between local and a remote server.
"""

import os
from dotenv import load_dotenv

# override=False means real env vars beat the .env file.
load_dotenv(override=False)

OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "granite3.3:2b")
