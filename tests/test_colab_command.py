"""Exercise Colab isolation and credential safety with a local fake CLI only."""

import json
import os
from pathlib import Path
import sys
import time

import pytest

from scripts import colab_command


SESSION = "book-recommendation-neural-test"


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    if os.name != "posix":
        pytest.skip("The Colab CLI process fixture requires POSIX executable scripts/signals.")
    executable = tmp_path / "fake-colab"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, subprocess, sys, time\n"
        "mode = os.environ.get('FAKE_COLAB_MODE', 'success')\n"
        "if mode == 'timeout':\n"
        "    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "    print('child_pid=' + str(child.pid), flush=True)\n"
        "    time.sleep(60)\n"
        "elif mode == 'traceback':\n"
        "    print('Traceback (most recent call last): remote ValueError')\n"
        "elif mode == 'exit':\n"
        "    print('remote rejected operation')\n"
        "    sys.exit(7)\n"
        "elif mode == 'credentials':\n"
        "    print('access_token=short-access refresh_token=short-refresh password=pass')\n"
        "    print('Authorization: Bearer abc-secret')\n"
        "    print('https://accounts.google.com/o/oauth2/auth?code=private')\n"
        "    sys.stdout.write('ya29.secret')\n"
        "    sys.stdout.flush()\n"
        "    time.sleep(0.03)\n"
        "    print('-access-value')\n"
        "else:\n"
        "    print(json.dumps({'argv': sys.argv[1:]}))\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    monkeypatch.setenv("COLAB_CLI", str(executable))
    monkeypatch.setenv("COLAB_OPERATIONS_DIR", str(tmp_path / "operations"))
    monkeypatch.setenv("COLAB_STATE_PATH", str(tmp_path / "state" / "sessions.json"))
    monkeypatch.setenv("COLAB_SESSION", SESSION)
    return executable


@pytest.mark.parametrize("command", sorted(colab_command.SESSION_COMMANDS))
def test_session_commands_are_explicitly_project_scoped(command):
    assert colab_command.with_session([command], SESSION) == [command, "-s", SESSION]


@pytest.mark.parametrize("arguments", [["--version"], ["--help"], ["sessions"], ["auth", "--help"]])
def test_global_flags_and_session_listing_do_not_receive_session(arguments):
    assert colab_command.with_session(arguments, SESSION) == arguments


@pytest.mark.parametrize("option", [["-s", "qlora-v3-portfolio"],
                                    ["--session", "qlora-v3-portfolio"],
                                    ["--session=qlora-v3-portfolio"],
                                    ["-sqlora-v3-portfolio"],
                                    ["--session", "book-recommendation-neural-other"]])
def test_cli_session_override_cannot_target_a_different_runtime(option):
    with pytest.raises(ValueError, match="exactly match"):
        colab_command.with_session(["stop", *option], SESSION)


@pytest.mark.parametrize("session", ["qlora-v3-portfolio", "../../escape", "", "book-recommendation-neural-"])
def test_nonproject_session_names_are_rejected(session):
    with pytest.raises(ValueError, match="beginning"):
        colab_command.with_session(["exec", "print(1)"], session)


def test_matching_session_is_allowed_and_remote_program_flags_are_preserved():
    args = ["exec", "--session=" + SESSION, "--", "program", "-s", "remote-option"]
    assert colab_command.with_session(args, SESSION) == args


@pytest.mark.parametrize("option", [["--config", "/tmp/other-project.json"],
                                    ["--config=/tmp/other-project.json"],
                                    ["--auth", "adc"], ["--auth=adc"]])
def test_auth_and_state_overrides_are_rejected(option):
    with pytest.raises(ValueError, match="fixed"):
        colab_command.with_session(["sessions", *option], SESSION)


def test_new_rejects_paid_gpu_and_permits_t4():
    with pytest.raises(ValueError, match="free-tier"):
        colab_command.with_session(["new", "--gpu", "A100"], SESSION)
    assert colab_command.with_session(["new", "--gpu=T4"], SESSION) == [
        "new", "-s", SESSION, "--gpu=T4",
    ]


def test_redaction_covers_credentials_oauth_urls_and_unlabeled_tokens():
    secrets = [
        "ya29.super-secret-access", "1//refresh-secret-value",
        "eyJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjoxfQ.secretSignature",
        "a" * 48, "quoted password value", "short-secret", "private-code",
    ]
    text = (
        f"{secrets[0]} {secrets[1]} {secrets[2]} {secrets[3]} "
        f'password="{secrets[4]}" secret={secrets[5]} '
        f"https://accounts.google.com/o/oauth2/auth?code={secrets[6]} "
        "Bearer tiny-token id_token='small-id' refresh-token: small-refresh"
    )
    redacted = colab_command.redact(text)
    for secret in [*secrets, "tiny-token", "small-id", "small-refresh"]:
        assert secret not in redacted
    assert "<redacted>" in redacted
    assert "<url-redacted>" in redacted


def test_plain_secret_argument_is_redacted():
    assert colab_command.redacted_arguments(["--token", "short-token", "status"]) == [
        "--token", "<redacted>", "status",
    ]


def test_command_uses_oauth_and_private_state_and_records_valid_json(fake_cli, tmp_path, capsys):
    assert colab_command.main(["safe", "--", "status"]) == 0
    record = json.loads((tmp_path / "operations" / "safe.json").read_text())
    argv = json.loads(record["combined_output"])["argv"]
    assert argv == ["--config", str(tmp_path / "state" / "sessions.json"),
                    "--auth", "oauth2", "status", "-s", SESSION]
    assert record["session"] == SESSION
    assert record["cli_exit_code"] == 0
    assert not record["timed_out"]
    assert not list((tmp_path / "operations").glob(".colab-operation-*"))
    assert "Operation log:" in capsys.readouterr().out


def test_credentials_never_reach_output_or_record(fake_cli, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_COLAB_MODE", "credentials")
    assert colab_command.main(["private", "--", "--version"]) == 0
    output = capsys.readouterr().out
    record_text = (tmp_path / "operations" / "private.json").read_text()
    for secret in ["short-access", "short-refresh", "abc-secret", "password=pass",
                   "ya29.secret-access-value", "accounts.google.com", "code=private"]:
        assert secret not in output
        assert secret not in record_text


def test_remote_traceback_changes_success_exit_to_failure(fake_cli, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_COLAB_MODE", "traceback")
    assert colab_command.main(["remote-error", "--", "exec", "print(1)"]) == 1
    record = json.loads((tmp_path / "operations" / "remote-error.json").read_text())
    assert record["reported_cell_error"]
    assert record["cli_exit_code"] == 0
    assert record["exit_code"] == 1


def test_cli_failure_code_is_preserved(fake_cli, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_COLAB_MODE", "exit")
    assert colab_command.main(["cli-error", "--", "status"]) == 7
    assert json.loads((tmp_path / "operations" / "cli-error.json").read_text())["cli_exit_code"] == 7


def test_timeout_terminates_cli_descendants_and_saves_partial_log(fake_cli, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_COLAB_MODE", "timeout")
    started = time.monotonic()
    assert colab_command.main(["--timeout", "0.3", "bounded", "--", "exec", "sleep"]) == 124
    assert time.monotonic() - started < 3
    record = json.loads((tmp_path / "operations" / "bounded.json").read_text())
    assert record["timed_out"]
    assert record["cli_exit_code"] < 0
    child_pid = int(record["combined_output"].split("child_pid=")[1].strip())
    status = Path(f"/proc/{child_pid}/stat")
    # An unreaped zombie has exited; it cannot continue consuming GPU resources.
    assert not status.exists() or status.read_text().split()[2] == "Z"


def test_logs_are_never_overwritten_and_unique_label_is_available(fake_cli, tmp_path):
    assert colab_command.main(["repeat", "--", "--version"]) == 0
    first = (tmp_path / "operations" / "repeat.json").read_bytes()
    with pytest.raises(SystemExit) as error:
        colab_command.main(["repeat", "--", "--version"])
    assert error.value.code == 2
    assert (tmp_path / "operations" / "repeat.json").read_bytes() == first
    assert colab_command.main(["--unique-label", "repeat", "--", "--version"]) == 0
    assert len(list((tmp_path / "operations").glob("repeat*.json"))) == 2


def test_launch_failure_is_recorded_without_leaking_token(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_OPERATIONS_DIR", str(tmp_path / "operations"))
    monkeypatch.setenv("COLAB_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("COLAB_CLI", str(tmp_path / "ya29.private-missing-cli"))
    assert colab_command.main(["failed-launch", "--", "--version"]) == 1
    record = json.loads((tmp_path / "operations" / "failed-launch.json").read_text())
    assert record["cli_exit_code"] is None
    assert record["launch_error"]
    assert "ya29.private-missing-cli" not in json.dumps(record)
