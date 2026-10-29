"""Download and verify immutable backups from this project's Colab worker.

This monitor never provisions or stops a runtime. Every operation uses the
project's OAuth2 helper and session state; the caller stops its own session
after all needed artifacts have been verified.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import time

try:
    from .colab_command import ROOT, validate_session
except ImportError:
    from colab_command import ROOT, validate_session


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_experiment_path(session, directory="experiment"):
    """Keep each run in one named directory under its owning project session."""
    validate_session(session)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", directory):
        raise ValueError("Experiment directory must be a simple name, not a path.")
    return f"/content/{session}/{directory}"


def invoke(session, label, arguments, timeout=180):
    environment = dict(os.environ)
    environment["COLAB_CLI"] = str(ROOT / "scripts/bounded_colab_cli.py")
    command = [sys.executable, str(ROOT / "scripts/colab_command.py"),
               "--session", session, "--timeout", str(timeout), "--unique-label",
               label, "--", *arguments]
    result = subprocess.run(command, env=environment, cwd=ROOT, capture_output=True,
                            text=True, timeout=timeout + 15)
    if result.returncode:
        # The helper already redacted every line and persisted its operation log.
        print(result.stdout[-1800:], flush=True)
        raise RuntimeError(f"Colab operation {label} failed: exit {result.returncode}")


def verify_and_extract(archive, directory, manifest):
    if archive.stat().st_size != manifest["bytes"]:
        raise RuntimeError(f"Backup size mismatch: {archive.name}")
    if digest_file(archive) != manifest["sha256"]:
        raise RuntimeError(f"Backup checksum mismatch: {archive.name}")
    # Tar readers may stop at tar's end marker before checking the gzip footer.
    with gzip.open(archive, "rb") as compressed:
        for _ in iter(lambda: compressed.read(1024 * 1024), b""):
            pass
    files = {}
    with tarfile.open(archive, "r:gz") as contents:
        members = contents.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise RuntimeError(f"Unsafe backup member: {member.name}")
            if member.isfile():
                # Record the expected extracted contents and check truncation.
                stream = contents.extractfile(member)
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                files[member.name] = digest.hexdigest()
        directory.mkdir(parents=True, exist_ok=True)
        contents.extractall(directory, filter="data")
    for name, expected in files.items():
        if digest_file(directory / name) != expected:
            raise RuntimeError(f"Extracted backup checksum mismatch: {name}")
    return {"archive": archive.name, "sha256": manifest["sha256"],
            "bytes": archive.stat().st_size, "members": len(members),
            "kind": manifest["kind"], "epoch": manifest.get("epoch"),
            "verified_utc": datetime.now(timezone.utc).isoformat(),
            "extracted_directory": str(directory), "files": files}


def cached_backup_valid(archive, directory, record, manifest):
    """Recheck local bytes before trusting verification from an earlier run."""
    if (not isinstance(record, dict) or "files" not in record
            or record.get("sha256") != manifest["sha256"]
            or record.get("bytes") != manifest["bytes"]):
        return False
    try:
        if (not archive.is_file() or archive.stat().st_size != manifest["bytes"]
                or digest_file(archive) != manifest["sha256"] or not directory.is_dir()):
            return False
        for name, expected in record["files"].items():
            relative = PurePosixPath(name)
            path = directory / name
            if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
                    or not path.resolve().is_relative_to(directory.resolve())
                    or not path.is_file() or digest_file(path) != expected):
                return False
    except (OSError, TypeError, AttributeError):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--local-output", type=Path, required=True)
    parser.add_argument("--experiment-directory", default="experiment")
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--max-seconds", type=float, default=1800)
    args = parser.parse_args()
    validate_session(args.session)
    if args.poll_seconds <= 0 or args.max_seconds <= 0:
        parser.error("Polling and total time limits must be positive.")
    remote = remote_experiment_path(args.session, args.experiment_directory)
    output = args.local_output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    control = output / "control"
    control.mkdir(exist_ok=True)
    poll_file = control / "poll_worker.py"
    poll_file.write_text(
        "from pathlib import Path\nimport json\n"
        f"output=Path({remote!r})\n"
        "record={'worker_exit_code':globals().get('_project_worker').poll() "
        "if globals().get('_project_worker') is not None else None}\n"
        "for name in ['progress.json','evaluation_progress.json','failed.json']:\n"
        "    if (output/name).exists(): record[name]=json.loads((output/name).read_text())\n"
        "record['backup_manifests']=[]\n"
        "for name in ['inputs_manifest.json', *[p.name for p in "
        "sorted(output.glob('epoch_*_manifest.json'))], 'final_manifest.json']:\n"
        "    if (output/name).exists():\n"
        "        record['backup_manifests'].append({'manifest_file':name,"
        "**json.loads((output/name).read_text())})\n"
        "record['completed']=(output/'completed.flag').exists()\n"
        "(output/'poll_status.json').write_text(json.dumps(record))\n"
        "print(json.dumps({'worker_exit_code':record['worker_exit_code'],"
        "'completed':record['completed'],'epoch':record.get('progress.json',{}).get('epoch'),"
        "'stage':record.get('progress.json',{}).get('status'),"
        "'backup_count':len(record['backup_manifests'])}))\n",
        encoding="utf-8",
    )
    verified_path = output / "backup_verification.json"
    verified = json.loads(verified_path.read_text()) if verified_path.exists() else {}
    started = time.monotonic()
    tick = 0
    while time.monotonic() - started < args.max_seconds:
        tick += 1
        invoke(args.session, "poll-experiment", ["exec", "--timeout", "45", "-f", str(poll_file)], 60)
        status_file = output / "status.json"
        invoke(args.session, "download-experiment-status",
               ["download", remote + "/poll_status.json", str(status_file)])
        status = json.loads(status_file.read_text())
        print(json.dumps({"tick": tick, "stage": status.get("progress.json", {}).get("status"),
                          "epoch": status.get("progress.json", {}).get("epoch"),
                          "completed": status["completed"]}), flush=True)
        for manifest in status["backup_manifests"]:
            name = manifest["archive"]
            if not re.fullmatch(r"[A-Za-z0-9._-]+", name) or not re.fullmatch(
                r"[a-f0-9]{64}", manifest["sha256"],
            ):
                raise RuntimeError("Unexpected backup manifest format.")
            if manifest["remote_path"] != remote + "/" + name:
                raise RuntimeError("Backup manifest points outside this project's output.")
            archive = output / name
            directory = output / ("final" if manifest["kind"] == "completed_experiment"
                                  else name.removesuffix(".tar.gz"))
            if cached_backup_valid(archive, directory, verified.get(name), manifest):
                continue
            invoke(args.session, "download-" + manifest["kind"],
                   ["download", manifest["remote_path"], str(archive)], 300)
            verified[name] = verify_and_extract(archive, directory, manifest)
            (output / (name + ".manifest.json")).write_text(json.dumps(manifest, indent=2))
            temporary = verified_path.with_suffix(".json.partial")
            temporary.write_text(json.dumps(verified, indent=2))
            temporary.replace(verified_path)
            print(json.dumps({"backup_verified": name, "epoch": manifest.get("epoch"),
                              "bytes": manifest["bytes"]}), flush=True)
        if status.get("failed.json"):
            invoke(args.session, "download-failed-worker-log",
                   ["download", remote + "/worker.log", str(output / "worker.log")])
            raise RuntimeError("Worker failed; local snapshots and worker log are preserved.")
        if status["completed"]:
            if "final_backup.tar.gz" not in verified:
                raise RuntimeError("Completed worker has no locally verified final backup.")
            print(json.dumps({"status": "completed", "all_backups_verified": True,
                              "local_output": str(output)}), flush=True)
            return
        if status["worker_exit_code"] is not None:
            invoke(args.session, "download-exited-worker-log",
                   ["download", remote + "/worker.log", str(output / "worker.log")])
            raise RuntimeError("Worker exited without a completion flag; inspect preserved log.")
        time.sleep(min(args.poll_seconds, max(0, args.max_seconds - (time.monotonic() - started))))
    raise TimeoutError("Experiment monitor reached its limit; inspect and back up before shutdown.")


if __name__ == "__main__":
    main()
