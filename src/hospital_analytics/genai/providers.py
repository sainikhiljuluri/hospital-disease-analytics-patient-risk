"""
MODULE 5, part 3: talking to a language model - or working without one.

FIVE WAYS TO GENERATE, IN ORDER OF PREFERENCE
----------------------------------------------
  1. OLLAMA      Llama running LOCALLY on your own machine   (free, no API key,
                 no data leaves the computer)  <-- THE PROJECT DEFAULT
  2. ANTHROPIC   Claude, via the `anthropic` package         (needs an API key)
  3. OPENAI      GPT, via the `openai` package                (needs an API key)
  4. HUGGINGFACE a local model via `transformers`             (free, but a big
                 download and slower than Ollama)
  5. TEMPLATE    no AI at all - fills the same structure from the same facts
                 using Python                                 (always available)

WHY OLLAMA AND LLAMA ARE THE DEFAULT
------------------------------------
Ollama runs Meta's Llama models on your own computer. For this project that is
not merely the cheap option - it is the CORRECT one, for a reason worth saying
out loud in a demo:

    **Patient data never leaves the machine.**

Sending real clinical records to a hosted API is a genuine legal problem under
GDPR and HIPAA, and "we pseudonymised the ID first" is a mitigation, not a
solution. A locally-hosted model removes the question entirely: there is no
outbound request, so there is nothing to leak, log, or subpoena.

It is also free, needs no account, no API key and no credit card, and it keeps
working when the venue wifi does not.

INSTALLING IT (about five minutes, one time)
--------------------------------------------
    1. Download Ollama from https://ollama.com/download  (Windows/Mac/Linux)
    2. Open a terminal and run:   ollama pull llama3.2
    3. That is it. Ollama runs a local server at http://localhost:11434
       and starts automatically with your computer.

This project talks to it using Python's BUILT-IN urllib - there is no extra
`pip install` at all. If Ollama is running, the project finds it.

CHOOSING A MODEL SIZE
---------------------
    ollama pull llama3.2         ~2 GB   default; fine on any modern laptop
    ollama pull llama3.2:1b      ~1.3 GB smaller and faster, weaker prose
    ollama pull llama3.1:8b      ~4.7 GB better writing, needs ~8 GB RAM

Set OLLAMA_MODEL in your .env file to switch. A risk summary is a short,
heavily-constrained writing task - the facts come from Module 2 and Module 4,
so a small model is genuinely sufficient here.

WHY OPTION 4 EXISTS, AND WHY IT IS NOT A CONSOLATION PRIZE
-----------------------------------------------------------
The template generator produces a complete, correct report with **no API key, no
internet, no downloads and no cost**. That matters for three practical reasons:

  * Your project must run on a marker's machine, which will not have your key.
  * API keys expire, run out of credit, and fail on demo day.
  * It proves a real point about the architecture: because everything is
    grounded in facts assembled by Modules 2 and 4, the facts are doing most of
    the work. The language model improves the PROSE; it does not supply the
    CONTENT.

That is the strongest thing you can say about this design. If swapping the AI
for a Python template still yields a correct report, the AI was never the source
of truth - which is exactly the property you want in a clinical tool.

Run both in your demo and show the difference. The LLM version reads better. The
template version is never wrong.

WHERE THE API KEY GOES
----------------------
In your `.env` file, never in code:

    ANTHROPIC_API_KEY=sk-ant-...
    # or
    OPENAI_API_KEY=sk-...

`.env` is already in `.gitignore`, so a key cannot be committed by accident.
"""

from __future__ import annotations

import os
from typing import Any

# Small, cheap, fast models. A risk summary is a short structured writing task -
# it does not need a frontier model, and using one would waste money.
# Ollama runs a local HTTP server. This is its default address; override with
# OLLAMA_HOST in .env if you changed the port.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# How long to wait for the local model to write a report. Generation on CPU is
# slow - 60 seconds is generous for a first run on a modest laptop, and the
# request is local so there is no network latency to worry about.
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

ANTHROPIC_MODEL = "claude-sonnet-4-6"
OPENAI_MODEL = "gpt-4o-mini"
HUGGINGFACE_MODEL = "google/flan-t5-base"


class GenerationError(RuntimeError):
    """Raised when a provider fails, with a readable explanation."""


# ---------------------------------------------------------------------------
def available_providers() -> dict[str, dict[str, Any]]:
    """Which providers can actually run right now, and why not if they cannot.

    The dashboard shows this table, so the reason a provider is unavailable is
    visible rather than mysterious.
    """
    providers: dict[str, dict[str, Any]] = {}

    # --- Ollama (local Llama) - checked first because it is the default ----
    ok, detail = ollama_status()
    providers["ollama"] = {
        "label": f"Ollama / Llama ({OLLAMA_MODEL}) - local, free, private",
        "ready": ok,
        "reason": detail,
        "needs_key": False,
        "local": True,
    }

    # --- Anthropic --------------------------------------------------------
    try:
        import anthropic  # noqa: F401
        has_package = True
    except ImportError:
        has_package = False
    has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    providers["anthropic"] = {
        "label": "Anthropic (Claude)",
        "ready": has_package and has_key,
        "reason": (
            "ready" if has_package and has_key
            else "package not installed - pip install anthropic" if not has_package
            else "no ANTHROPIC_API_KEY in your .env file"
        ),
        "needs_key": True,
    }

    # --- OpenAI -----------------------------------------------------------
    try:
        import openai  # noqa: F401
        has_package = True
    except ImportError:
        has_package = False
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    providers["openai"] = {
        "label": "OpenAI (GPT)",
        "ready": has_package and has_key,
        "reason": (
            "ready" if has_package and has_key
            else "package not installed - pip install openai" if not has_package
            else "no OPENAI_API_KEY in your .env file"
        ),
        "needs_key": True,
    }

    # --- Hugging Face -----------------------------------------------------
    try:
        import transformers  # noqa: F401
        has_package = True
    except ImportError:
        has_package = False
    providers["huggingface"] = {
        "label": "Hugging Face (local model, free)",
        "ready": has_package,
        "reason": (
            "ready - first run downloads ~1 GB" if has_package
            else "package not installed - pip install transformers torch"
        ),
        "needs_key": False,
    }

    # --- Template ---------------------------------------------------------
    providers["template"] = {
        "label": "Template (no AI, always works)",
        "ready": True,
        "reason": "always available - no key, no internet, no downloads",
        "needs_key": False,
    }
    return providers


def best_available() -> str:
    """Pick the best provider that can actually run.

    Ollama comes FIRST deliberately. It is free, needs no key, and - the reason
    that actually matters for clinical data - it keeps every patient record on
    the local machine. A hosted API is only preferred if you explicitly ask for
    one by name.

    An explicit GENAI_PROVIDER in .env overrides this ordering entirely.
    """
    forced = os.getenv("GENAI_PROVIDER", "auto").strip().lower()
    if forced and forced != "auto":
        return forced

    providers = available_providers()
    for name in ("ollama", "anthropic", "openai", "huggingface"):
        if providers.get(name, {}).get("ready"):
            return name
    return "template"


# ---------------------------------------------------------------------------
# OLLAMA - a local Llama model
# ---------------------------------------------------------------------------
def _ollama_request(path: str, payload: dict | None = None, timeout: int = 5) -> Any:
    """Call the local Ollama server using only the Python standard library.

    Deliberately NOT using `requests` or the `ollama` package. Ollama exposes a
    plain HTTP API, and urllib is built into Python - so this provider adds
    ZERO new dependencies to the project. One less thing to install, and one
    less thing to break on a marker's machine.
    """
    import json
    import urllib.error
    import urllib.request

    url = f"{OLLAMA_HOST.rstrip('/')}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_status() -> tuple[bool, str]:
    """Is Ollama running, and is the requested model pulled?

    Returns (ready, human-readable reason). Every failure mode gets its own
    message with the exact command that fixes it - a status line saying only
    "not available" would send a beginner hunting through documentation.
    """
    import urllib.error

    try:
        payload = _ollama_request("/api/tags", timeout=3)
    except urllib.error.URLError:
        return False, (
            f"Ollama is not running at {OLLAMA_HOST}. "
            "Install it from https://ollama.com/download, then run: "
            f"ollama pull {OLLAMA_MODEL}"
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"Could not reach Ollama at {OLLAMA_HOST} ({type(exc).__name__})."

    installed = [m.get("name", "") for m in payload.get("models", [])]
    if not installed:
        return False, (
            f"Ollama is running but has no models. Run: ollama pull {OLLAMA_MODEL}"
        )

    # Ollama reports names as "llama3.2:latest"; accept a bare "llama3.2".
    wanted = OLLAMA_MODEL.split(":")[0]
    if any(name.split(":")[0] == wanted for name in installed):
        return True, f"ready - {OLLAMA_MODEL} running locally, no data leaves this machine"

    return False, (
        f"Ollama is running but '{OLLAMA_MODEL}' is not installed. "
        f"Run: ollama pull {OLLAMA_MODEL}   "
        f"(installed: {', '.join(installed[:3]) or 'none'})"
    )


def _generate_ollama(system: str, user: str, max_tokens: int = 1200) -> str:
    """Generate with a local Llama model through Ollama's chat API.

    `temperature` is set low on purpose. This is a factual clinical summary
    assembled from supplied numbers, not creative writing - we want the model to
    stay close to the facts it was given rather than embellish them.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,     # low: stick to the supplied facts
            "top_p": 0.9,
            "num_predict": max_tokens,
        },
    }
    result = _ollama_request("/api/chat", payload, timeout=OLLAMA_TIMEOUT)
    text = (result.get("message") or {}).get("content", "")
    if not text:
        raise GenerationError(
            f"Ollama returned an empty response. Check that '{OLLAMA_MODEL}' is "
            f"pulled and that the model finished loading."
        )
    return str(text).strip()


# ---------------------------------------------------------------------------
def _generate_anthropic(system: str, user: str, max_tokens: int = 1200) -> str:
    import anthropic

    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise GenerationError("ANTHROPIC_API_KEY is not set in your .env file.")

    client = anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()


def _generate_openai(system: str, user: str, max_tokens: int = 1200) -> str:
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise GenerationError("OPENAI_API_KEY is not set in your .env file.")

    client = OpenAI(api_key=key)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _generate_huggingface(system: str, user: str, max_tokens: int = 1200) -> str:
    """Run a small model locally. Free, but slower and noticeably weaker.

    Small instruction-tuned models struggle with long structured prompts, so
    output quality varies. Included because it is the only way to use a real
    language model with no API key and no data leaving the machine — which for
    genuine patient data would be the only acceptable option.
    """
    from transformers import pipeline

    generator = pipeline("text2text-generation", model=HUGGINGFACE_MODEL)
    combined = f"{system}\n\n{user}"
    result = generator(combined, max_new_tokens=min(max_tokens, 512))
    return str(result[0].get("generated_text", "")).strip()


PROVIDER_FUNCTIONS = {
    "ollama": _generate_ollama,
    "anthropic": _generate_anthropic,
    "openai": _generate_openai,
    "huggingface": _generate_huggingface,
}


def generate(
    system: str, user: str, provider: str = "auto", max_tokens: int = 1200
) -> dict[str, Any]:
    """Send a prompt to a provider and return the text plus what happened.

    Never raises on provider failure. If the API is down, out of credit or
    misconfigured, this returns `used_fallback=True` and the caller falls back to
    the template generator. A demo should not die because a network call failed.
    """
    if provider == "auto":
        provider = best_available()

    if provider == "template":
        return {"text": "", "provider": "template", "used_fallback": False, "error": None}

    function = PROVIDER_FUNCTIONS.get(provider)
    if function is None:
        return {
            "text": "", "provider": "template", "used_fallback": True,
            "error": f"Unknown provider '{provider}'.",
        }

    try:
        text = function(system, user, max_tokens=max_tokens)
        if not text:
            raise GenerationError("The provider returned an empty response.")
        return {"text": text, "provider": provider, "used_fallback": False, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {
            "text": "", "provider": "template", "used_fallback": True,
            "error": f"{type(exc).__name__}: {exc}",
        }
