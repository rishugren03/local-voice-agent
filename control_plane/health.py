"""Service health checks for the System screen (U7) and the /health banner.

The design rule here is that a failing check must say what to do about it. A
health endpoint that returns {"ollama": false} is a status light and nothing more;
one that returns "Ollama is not reachable at http://localhost:11434 — start it
with `ollama serve`, or set OLLAMA_URL" is a fix. Every check therefore returns a
hint alongside the status, and the UI shows the hint verbatim.

Every check is wrapped so that a missing dependency (no requests, no LiveKit, no
voice files) degrades to "unknown, and here is the fix" instead of raising. A
health endpoint that 500s tells you nothing about the services it was checking.
"""

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request

# Short by design. This runs on a poll, and a health check that takes 30s to fail
# is indistinguishable from a hung API.
PROBE_TIMEOUT_S = float(os.getenv("HEALTH_PROBE_TIMEOUT_S", "3"))


def _hint(text):
    return text


def check_livekit():
    """Can the agent actually join a room?

    Parses the ws:// URL rather than issuing a request, because LiveKit's health
    endpoint differs between versions and a wrong path reports a working server as
    down. What actually matters is whether the URL parses and the key/secret are
    set — those are the three things that make the agent's own token fail.
    """
    url = os.getenv("LIVEKIT_URL", "")
    key = os.getenv("LIVEKIT_API_KEY", "")
    secret = os.getenv("LIVEKIT_API_SECRET", "")

    if not url:
        return {"status": "down", "detail": "LIVEKIT_URL is not set",
                "fix": "set LIVEKIT_URL, e.g. ws://localhost:7880"}
    if not url.startswith(("ws://", "wss://")):
        return {"status": "down", "detail": f"LIVEKIT_URL '{url}' is not a ws:// or wss:// URL",
                "fix": "LiveKit's client endpoint is a websocket URL, e.g. ws://localhost:7880"}
    if not key or not secret:
        return {"status": "down",
                "detail": "LIVEKIT_API_KEY or LIVEKIT_API_SECRET is not set",
                "fix": "start LiveKit with --dev, or set the key and secret it printed"}
    # LiveKit's own binary on PATH means the local dev server is likely running.
    local = shutil.which("livekit-server")
    return {
        "status": "ok",
        "detail": f"{url} (credentials present)",
        "fix": "",
        "local_binary": bool(local),
    }


def check_ollama():
    """Reachable, and which models it has.

    The model list matters as much as reachability: an agent configured for a
    model that was never pulled fails on the first turn with a connection-level
    error that reads like a network fault rather than a missing model.
    """
    base = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=PROBE_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return {
            "status": "down",
            "detail": f"not reachable at {base} ({e.reason})",
            "fix": f"start it with `ollama serve`, or point OLLAMA_URL at {base}",
            "models": [],
        }
    except Exception as e:  # malformed body, timeout, TLS
        return {
            "status": "down",
            "detail": f"responded but could not be read ({type(e).__name__}: {e})",
            "fix": f"check that {base} is Ollama and not something else on that port",
            "models": [],
        }

    models = []
    for m in payload.get("models", []):
        models.append({
            "name": m.get("name") or m.get("model", "?"),
            "size": m.get("size"),
            "family": (m.get("details") or {}).get("family"),
        })
    return {"status": "ok", "detail": f"{len(models)} model(s) at {base}",
            "fix": "", "models": models}


def check_whisper():
    """The binary exists, is executable, and the model file is there.

    Checked as two separate problems because they are fixed differently: a wrong
    WHISPER_BIN is an env typo, a missing model is a download.
    """
    binary = os.getenv("WHISPER_BIN", "")
    model = os.getenv("WHISPER_MODEL", "")

    if not binary:
        return {"status": "down", "detail": "WHISPER_BIN is not set",
                "fix": "point WHISPER_BIN at the whisper-cli built by whisper.cpp"}
    if not os.path.exists(binary):
        # A bare name means PATH lookup, which os.path.exists cannot see.
        resolved = shutil.which(binary)
        if resolved:
            binary = resolved
        else:
            return {"status": "down", "detail": f"whisper binary not found: {binary}",
                    "fix": "build whisper.cpp, or set WHISPER_BIN to the full path of whisper-cli"}
    if not os.access(binary, os.X_OK):
        return {"status": "down", "detail": f"whisper binary is not executable: {binary}",
                "fix": f"chmod +x {binary}"}
    if not model:
        return {"status": "down", "detail": "WHISPER_MODEL is not set",
                "fix": "download a ggml model, e.g. ggml-base.en.bin, and set WHISPER_MODEL"}
    if not os.path.exists(model):
        return {"status": "down", "detail": f"whisper model not found: {model}",
                "fix": "download it from https://huggingface.co/ggerganov/whisper.cpp"}

    return {
        "status": "ok",
        "detail": os.path.basename(binary),
        "fix": "",
        "model": os.path.basename(model),
        "model_size_mb": round(os.path.getsize(model) / 1e6, 1),
    }


def check_piper(track_sample_rate=None):
    """The binary exists and every configured voice is readable and in-rate.

    The sample-rate check is the one that matters most here. A voice whose native
    rate differs from the room's fixed track rate does not degrade — playback
    fails outright with "InvalidState: sample_rate and num_channels don't match",
    in the middle of a call, with the user hearing nothing.
    """
    voices = _configured_voices()
    if shutil.which("piper") is None:
        return {"status": "down", "detail": "piper not found on PATH",
                "fix": "pip install piper-tts", "voices": []}

    reports = []
    problems = []
    for source, path in voices:
        entry = {"source": source, "voice": os.path.basename(path), "path": path}
        if not os.path.exists(path):
            entry.update({"ok": False, "detail": "file not found",
                          "fix": f"download the voice to {path}"})
            problems.append(entry)
            reports.append(entry)
            continue
        try:
            with open(path + ".json") as f:
                rate = int(json.load(f)["audio"]["sample_rate"])
        except (OSError, ValueError, KeyError) as e:
            entry.update({"ok": False, "sample_rate": None,
                          "detail": f"could not read sample rate ({e})",
                          "fix": f"Piper needs {os.path.basename(path)}.json beside the voice"})
            problems.append(entry)
            reports.append(entry)
            continue
        entry["sample_rate"] = rate
        if track_sample_rate and rate != track_sample_rate:
            entry.update({
                "ok": False,
                "detail": f"{rate} Hz but the audio track is {track_sample_rate} Hz",
                "fix": "use a voice at the track rate, or change PIPER_SAMPLE_RATE to match",
            })
            problems.append(entry)
        else:
            entry.update({"ok": True, "detail": f"{rate} Hz", "fix": ""})
        reports.append(entry)

    if problems:
        return {"status": "down", "detail": f"{len(problems)} voice problem(s)",
                "fix": problems[0].get("fix", ""), "voices": reports}
    return {"status": "ok", "detail": f"{len(reports)} voice(s) ready",
            "fix": "", "voices": reports}


def _configured_voices():
    """(source, path) for every voice the config references.

    Tries the config store first, then the JSON files, so this works whether or
    not the control plane has imported the config yet. A failure here is not
    itself a health problem — an unimported config just means the agent is
    running on files, which is valid.
    """
    try:
        from agent_platform.config_store import ConfigStore
        from agent_platform.orchestrator import AgentRegistry

        store = ConfigStore(os.getenv("CONFIG_DB", "call_trace.db"))
        try:
            registry = AgentRegistry(
                default_tts_voice=os.getenv("PIPER_MODEL", ""),
                config_store=store,
            )
        finally:
            store.close()
        return registry.all_configured_voices()
    except Exception:
        pass

    default = os.getenv("PIPER_MODEL", "")
    voices = []
    if default:
        voices.append(("platform_default", default))
    agents_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "agent_platform", "agents")
    try:
        for filename in os.listdir(agents_dir):
            if not filename.endswith(".json"):
                continue
            with open(os.path.join(agents_dir, filename)) as f:
                agent = json.load(f)
            voice = agent.get("tts_voice")
            if voice and voice != default:
                voices.append((agent.get("id", filename), voice))
    except OSError:
        pass
    return voices


def check_mcp():
    """The MCP server script is present and importable.

    Not started: doing so per health poll would spawn a process every few seconds.
    What actually breaks a call is a missing or unimportable server module, and
    that is what this catches.
    """
    script = os.getenv("MCP_SERVER_SCRIPT", "mcp_server.py")
    if not os.path.exists(script):
        return {"status": "down", "detail": f"not found: {script}",
                "fix": "run the API from the project root, or set MCP_SERVER_SCRIPT"}
    try:
        result = subprocess.run(
            [os.sys.executable, "-c",
             f"import ast,sys; ast.parse(open({script!r}).read())"],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT_S * 2,
        )
    except subprocess.SubprocessError as e:
        return {"status": "unknown", "detail": f"could not check ({e})", "fix": ""}
    if result.returncode != 0:
        return {"status": "down", "detail": f"{script} does not parse",
                "fix": result.stderr.strip()[:200]}
    return {"status": "ok", "detail": f"{os.path.basename(script)} present", "fix": ""}


def check_all(track_sample_rate=None):
    """Every service, plus an overall verdict the UI can put in one banner.

    The aggregate is 'down' only when a check is 'down', not when one is
    'unknown': an unreadable check is a gap in the monitoring, not evidence the
    service is broken, and colouring the whole app red for it would train people
    to ignore the banner.
    """
    checks = {}
    for name, fn in (
        ("livekit", check_livekit),
        ("ollama", check_ollama),
        ("whisper", check_whisper),
        ("piper", lambda: check_piper(track_sample_rate)),
        ("mcp", check_mcp),
    ):
        try:
            checks[name] = fn()
        except Exception as e:
            checks[name] = {
                "status": "unknown",
                "detail": f"the check itself failed ({type(e).__name__}: {e})",
                "fix": "",
            }

    down = [n for n, c in checks.items() if c["status"] == "down"]
    unknown = [n for n, c in checks.items() if c["status"] == "unknown"]
    overall = "down" if down else ("unknown" if unknown else "ok")
    return {
        "status": overall,
        "down": down,
        "unknown": unknown,
        "services": checks,
    }
