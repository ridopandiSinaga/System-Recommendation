"""Run a bounded Colab CLI operation with project-isolated, credential-safe logs.

Usage: python3 scripts/colab_command.py [--session NAME] [--timeout SECONDS]
       [--unique-label] LABEL -- COMMAND [ARGS...]

COLAB_CLI overrides the executable; COLAB_SESSION supplies the default project
session; COLAB_OPERATIONS_DIR overrides artifacts/colab/operations and
COLAB_STATE_PATH overrides artifacts/colab/sessions.json. Only sessions
named book-recommendation-neural-* are permitted. Calls always use OAuth2.
This helper creates no runtime by itself and never changes another project's
runtime. Choose free-tier hardware when invoking the CLI's new command.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLI = "colab"
SESSION_PREFIX = "book-recommendation-neural-"
DEFAULT_SESSION = SESSION_PREFIX + "colab"
SESSION_COMMANDS = {
    "new", "status", "exec", "upload", "download", "stop", "install", "ls", "rm",
    "url", "log", "restart-kernel", "drivemount", "edit", "console", "repl", "run",
}
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}\Z")
SECRET_FIELD = re.compile(
    r"(?i)([\"']?\b(?:access[_-]?token|refresh[_-]?token|id[_-]?token|"
    r"api[_-]?key|authorization|password|secret|oauth[_-]?code|token)\b"
    r"[\"']?\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;\]}]+)"
)
URL = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
BEARER = re.compile(r"(?i)\bBearer\s+[^\s,;\]}\"']+")
RAW_TOKEN = re.compile(
    r"(?:ya29\.[A-Za-z0-9._~-]+|1//[A-Za-z0-9._~-]+|"
    r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+|"
    r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{40,}(?![A-Za-z0-9_-]))"
)
SECRET_OPTION = re.compile(
    r"--(?:access[-_]token|refresh[-_]token|id[-_]token|api[-_]key|"
    r"authorization|password|secret|oauth[-_]code|token)\Z", re.IGNORECASE
)


def redact(value):
    """Remove credentials, token-shaped strings, and OAuth URLs before output."""
    value = URL.sub("<url-redacted>", str(value))
    value = BEARER.sub("Bearer <redacted>", value)
    value = SECRET_FIELD.sub(r"\1<redacted>", value)
    return RAW_TOKEN.sub("<redacted>", value)


def redacted_arguments(arguments):
    result = []
    hide_next = False
    for argument in arguments:
        result.append("<redacted>" if hide_next else redact(argument))
        hide_next = bool(SECRET_OPTION.fullmatch(argument))
    return result


def validate_session(session):
    if (not SAFE_NAME.fullmatch(session) or not session.startswith(SESSION_PREFIX)
            or session == SESSION_PREFIX):
        raise ValueError(f"Session must be a simple name beginning with {SESSION_PREFIX}.")


def with_session(arguments, session):
    """Inject the selected project session and reject alternate CLI overrides."""
    validate_session(session)
    if not arguments:
        raise ValueError("A Colab subcommand or global flag is required.")
    found = []
    for index, argument in enumerate(arguments):
        if argument == "--":
            break  # Remaining values are a remote program's arguments.
        if argument in {"-s", "--session"}:
            if index + 1 >= len(arguments):
                raise ValueError("Session option requires a value.")
            found.append(arguments[index + 1])
        elif argument.startswith("--session="):
            found.append(argument.partition("=")[2])
        elif argument.startswith("-s") and not argument.startswith("--") and len(argument) > 2:
            found.append(argument[2:])
        if argument in {"--auth", "--auth-mode", "--config"} or argument.startswith(
                ("--auth=", "--auth-mode=", "--config=")):
            raise ValueError("Authentication and project state are fixed by this helper.")
    if any(value != session for value in found):
        raise ValueError("CLI session override must exactly match the selected project session.")
    if arguments[0] == "new":
        for index, argument in enumerate(arguments):
            if argument == "--gpu":
                if index + 1 >= len(arguments):
                    raise ValueError("GPU option requires a value.")
                gpu = arguments[index + 1]
            elif argument.startswith("--gpu="):
                gpu = argument.partition("=")[2]
            else:
                continue
            if gpu.lower() not in {"t4", "none", ""}:
                raise ValueError("Only free-tier T4 or a CPU runtime is permitted.")
    if arguments[0] in SESSION_COMMANDS and not found:
        return [arguments[0], "-s", session, *arguments[1:]]
    return list(arguments)


def _stop_process_group(process):
    """Terminate the local CLI and descendants without affecting other sessions."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        if sig == signal.SIGTERM:
            # Always send KILL afterward: a descendant may survive its parent.
            time.sleep(0.2)
    process.wait(timeout=5)


def run_command(command, timeout):
    started = datetime.now(timezone.utc).isoformat()
    started_clock = time.monotonic()
    lines = []
    process = None
    launch_error = None
    timed_out = False
    interrupted = False
    pending = b""

    def emit(raw):
        line = redact(raw.decode("utf-8", errors="replace").replace("\x00", ""))
        lines.append(line)
        print(line.rstrip().encode("ascii", "backslashreplace").decode(), flush=True)

    try:
        process = subprocess.Popen(
            command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            eof = False
            while not eof or process.poll() is None:
                remaining = timeout - (time.monotonic() - started_clock)
                if remaining <= 0:
                    timed_out = True
                    _stop_process_group(process)
                    break
                for key, _ in selector.select(min(remaining, 0.2)):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        eof = True
                        continue
                    pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        emit(line + b"\n")
            if process.poll() is None:
                process.wait(timeout=max(0.01, timeout - (time.monotonic() - started_clock)))
    except KeyboardInterrupt:
        interrupted = True
        if process is not None:
            _stop_process_group(process)
    except (OSError, subprocess.SubprocessError) as error:
        launch_error = redact(f"{type(error).__name__}: {error}")
        print(launch_error, file=sys.stderr, flush=True)
        if process is not None:
            _stop_process_group(process)
    finally:
        if pending:
            emit(pending)
        if process is not None and process.stdout is not None:
            process.stdout.close()

    combined_output = "".join(lines)
    reported_cell_error = "traceback" in combined_output.lower()
    cli_exit_code = process.returncode if process is not None else None
    exit_code = (130 if interrupted else 124 if timed_out else 1
                 if launch_error or reported_cell_error else cli_exit_code)
    return {
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started_clock, 3),
        "timeout_seconds": timeout,
        "timed_out": timed_out,
        "interrupted": interrupted,
        "cli_exit_code": cli_exit_code,
        "reported_cell_error": reported_cell_error,
        "exit_code": exit_code,
        "launch_error": launch_error,
        "combined_output": combined_output,
    }


def write_record(path, record):
    """Atomically publish a JSON log, refusing to replace an existing record."""
    descriptor, temporary = tempfile.mkstemp(prefix=".colab-operation-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default=os.environ.get("COLAB_SESSION", DEFAULT_SESSION))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--unique-label", action="store_true")
    parser.add_argument("label")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not SAFE_NAME.fullmatch(args.label) or args.label in {".", ".."}:
        parser.error("LABEL must be a simple operation name.")
    if not 0 < args.timeout < float("inf"):
        parser.error("Timeout must be a finite positive number of seconds.")
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    try:
        arguments = with_session(arguments, args.session)
    except ValueError as error:
        parser.error(str(error))

    operations = Path(os.environ.get("COLAB_OPERATIONS_DIR", ROOT / "artifacts/colab/operations"))
    operations.mkdir(parents=True, exist_ok=True)
    label = args.label
    if args.unique_label:
        label += "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    path = operations / f"{label}.json"
    if path.exists():
        parser.error(f"Operation log already exists: {path}; use --unique-label to preserve it.")
    state_path = Path(os.environ.get("COLAB_STATE_PATH", ROOT / "artifacts/colab/sessions.json"))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    command = [os.environ.get("COLAB_CLI", DEFAULT_CLI), "--config", str(state_path),
               "--auth", "oauth2", *arguments]
    record = run_command(command, args.timeout)
    record.update({
        "label": label,
        "arguments": redacted_arguments(arguments),
        "command": redacted_arguments(command),
        "session": args.session,
        "state_path": str(state_path),
    })
    write_record(path, record)
    print(f"Operation log: {path}", flush=True)
    return record["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
