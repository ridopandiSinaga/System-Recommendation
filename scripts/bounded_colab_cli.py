#!/usr/bin/env python3
"""Colab CLI 0.6.0 entrypoint with bounded file-transfer HTTP inactivity.

Use only through scripts/colab_command.py via COLAB_CLI for this project.
No retries, proxy refreshes, credential changes, or persistent CLI patches.
"""
from contextvars import ContextVar
from functools import wraps
from importlib.metadata import version
import sys

TRANSFER_TIMEOUT = (90, 180)
_active_transfer = ContextVar('colab_bounded_file_transfer', default=False)
_installed = False


class _RequestsProxy:
    def __init__(self, upstream):
        self._upstream = upstream

    def request(self, *args, **kwargs):
        if _active_transfer.get():
            kwargs.setdefault('timeout', TRANSFER_TIMEOUT)
        return self._upstream.request(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._upstream, name)


def _transfer_only(method):
    @wraps(method)
    def wrapped(*args, **kwargs):
        marker = _active_transfer.set(True)
        try:
            return method(*args, **kwargs)
        finally:
            _active_transfer.reset(marker)
    return wrapped


def install_bounded_file_transport():
    global _installed
    assert version('google-colab-cli') == '0.6.0', 'Review transport hook before changing CLI version.'
    if _installed:
        return
    from colab_cli import contents
    contents.requests = _RequestsProxy(contents.requests)
    contents.ContentsClient.upload = _transfer_only(contents.ContentsClient.upload)
    contents.ContentsClient.download = _transfer_only(contents.ContentsClient.download)
    _installed = True


def main():
    install_bounded_file_transport()
    from colab_cli.cli import main as upstream_main
    return upstream_main()


if __name__ == '__main__':
    sys.exit(main())
