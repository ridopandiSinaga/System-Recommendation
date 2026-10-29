"""Verify local backup integrity and containment without contacting Colab."""

import gzip
import hashlib
import io
import shutil
import tarfile

import pytest

from scripts import monitor_colab_experiment as monitor


def test_separate_runs_remain_within_the_owning_session():
    session = "book-recommendation-neural-example"
    assert monitor.remote_experiment_path(session) == f"/content/{session}/experiment"
    assert monitor.remote_experiment_path(session, "warmup_experiment") == (
        f"/content/{session}/warmup_experiment"
    )


@pytest.mark.parametrize("directory", ["../other", "/content/other", "a/b", "..", "", "x" * 81])
def test_run_directory_cannot_select_a_path_outside_session(directory):
    with pytest.raises(ValueError, match="simple name"):
        monitor.remote_experiment_path("book-recommendation-neural-example", directory)


def make_archive(path, member_name="checkpoint/weights.bin", *, link=None):
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo(member_name)
        if link is not None:
            member.type, member.linkname = link
            archive.addfile(member)
        else:
            payload = b"checkpoint weights\n"
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "kind": "epoch_checkpoint",
        "epoch": 2,
    }


def test_verified_backup_extracts_checkpoint_and_records_evidence(tmp_path):
    archive = tmp_path / "epoch_002_snapshot.tar.gz"
    manifest = make_archive(archive)
    destination = tmp_path / "restored"

    evidence = monitor.verify_and_extract(archive, destination, manifest)

    assert (destination / "checkpoint/weights.bin").read_bytes() == b"checkpoint weights\n"
    assert evidence["archive"] == archive.name
    assert evidence["sha256"] == manifest["sha256"]
    assert evidence["bytes"] == manifest["bytes"]
    assert evidence["members"] == 1
    assert evidence["epoch"] == 2
    assert evidence["verified_utc"]


@pytest.mark.parametrize("change", ["truncated", "same_size_corruption"])
def test_modified_download_is_rejected_before_extraction(tmp_path, change):
    archive = tmp_path / "snapshot.tar.gz"
    manifest = make_archive(archive)
    payload = archive.read_bytes()
    if change == "truncated":
        archive.write_bytes(payload[:-1])
        message = "size mismatch"
    else:
        archive.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
        message = "checksum mismatch"
    destination = tmp_path / "restored"

    with pytest.raises(RuntimeError, match=message):
        monitor.verify_and_extract(archive, destination, manifest)

    assert not destination.exists()


@pytest.mark.parametrize("member_name", ["../escaped.bin", "/absolute.bin", "ok/../../escaped.bin"])
def test_backup_paths_cannot_escape_destination(tmp_path, member_name):
    archive = tmp_path / "snapshot.tar.gz"
    manifest = make_archive(archive, member_name)
    destination = tmp_path / "restored"

    with pytest.raises(RuntimeError, match="Unsafe backup member"):
        monitor.verify_and_extract(archive, destination, manifest)

    assert not destination.exists()
    assert not (tmp_path / "escaped.bin").exists()


@pytest.mark.parametrize("link_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_backup_links_cannot_refer_to_files_outside_destination(tmp_path, link_type):
    archive = tmp_path / "snapshot.tar.gz"
    manifest = make_archive(archive, "checkpoint/link", link=(link_type, "../../private.bin"))
    destination = tmp_path / "restored"

    with pytest.raises(RuntimeError, match="Unsafe backup member"):
        monitor.verify_and_extract(archive, destination, manifest)

    assert not destination.exists()


def test_invalid_archive_is_rejected_even_with_matching_manifest(tmp_path):
    archive = tmp_path / "snapshot.tar.gz"
    archive.write_bytes(b"not a compressed archive")
    manifest = {
        "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "kind": "completed_experiment",
    }
    destination = tmp_path / "restored"

    with pytest.raises((gzip.BadGzipFile, tarfile.ReadError)):
        monitor.verify_and_extract(archive, destination, manifest)

    assert not destination.exists()


def test_corrupt_gzip_footer_is_rejected_even_with_matching_manifest(tmp_path):
    archive = tmp_path / "snapshot.tar.gz"
    manifest = make_archive(archive)
    payload = bytearray(archive.read_bytes())
    payload[-8] ^= 1  # Corrupt gzip CRC while leaving readable tar members intact.
    archive.write_bytes(payload)
    manifest["sha256"] = hashlib.sha256(payload).hexdigest()
    destination = tmp_path / "restored"

    with pytest.raises((gzip.BadGzipFile, tarfile.ReadError)):
        monitor.verify_and_extract(archive, destination, manifest)

    assert not destination.exists()


def test_verified_cache_requires_current_archive_and_extracted_content(tmp_path):
    archive = tmp_path / "snapshot.tar.gz"
    manifest = make_archive(archive)
    destination = tmp_path / "restored"
    evidence = monitor.verify_and_extract(archive, destination, manifest)

    assert monitor.cached_backup_valid(archive, destination, evidence, manifest)


@pytest.mark.parametrize("change", [
    "archive_missing", "archive_corrupt", "destination_missing",
    "checkpoint_missing", "checkpoint_corrupt", "manifest_changed",
])
def test_cached_evidence_cannot_hide_lost_or_changed_backup(tmp_path, change):
    archive = tmp_path / "snapshot.tar.gz"
    manifest = make_archive(archive)
    destination = tmp_path / "restored"
    evidence = monitor.verify_and_extract(archive, destination, manifest)
    checkpoint = destination / "checkpoint/weights.bin"
    if change == "archive_missing":
        archive.unlink()
    elif change == "archive_corrupt":
        payload = archive.read_bytes()
        archive.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    elif change == "destination_missing":
        shutil.rmtree(destination)
    elif change == "checkpoint_missing":
        checkpoint.unlink()
    elif change == "checkpoint_corrupt":
        checkpoint.write_bytes(b"modified checkpoint\n")
    elif change == "manifest_changed":
        manifest["sha256"] = "0" * 64

    assert not monitor.cached_backup_valid(archive, destination, evidence, manifest)


def test_legacy_cache_without_extracted_content_hashes_requires_revalidation(tmp_path):
    archive = tmp_path / "snapshot.tar.gz"
    manifest = make_archive(archive)
    destination = tmp_path / "restored"
    evidence = monitor.verify_and_extract(archive, destination, manifest)
    evidence.pop("files")

    assert not monitor.cached_backup_valid(archive, destination, evidence, manifest)


def test_cached_evidence_rejects_checkpoint_replaced_by_external_symlink(tmp_path):
    archive = tmp_path / "snapshot.tar.gz"
    manifest = make_archive(archive)
    destination = tmp_path / "restored"
    evidence = monitor.verify_and_extract(archive, destination, manifest)
    checkpoint = destination / "checkpoint/weights.bin"
    outside = tmp_path / "outside.bin"
    outside.write_bytes(checkpoint.read_bytes())
    checkpoint.unlink()
    try:
        checkpoint.symlink_to(outside)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Creating symlinks requires Windows Developer Mode or elevated privileges.")
        raise

    assert not monitor.cached_backup_valid(archive, destination, evidence, manifest)
