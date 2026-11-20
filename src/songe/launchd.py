"""Legacy service entry points; implementation is in songe.platform.services."""

import os
import re
import subprocess
import sys

from .platform import current
from .platform.services import Launchd


def stable_python(path=None):
    """Homebrew's opt link survives upgrades; the versioned Cellar path does not."""
    path = path or sys.executable
    found = re.match(r"(.*)/Cellar/([^/]+)/[^/]+/(.*)", path)
    if found:
        opt = f"{found[1]}/opt/{found[2]}/{found[3]}"
        if os.path.exists(opt):
            return opt
    return path


def arguments(root):
    return [stable_python(), "-m", "songe", "--data-dir", str(root), "serve"]


def environment():
    return {"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"} | {
        k: os.environ[k] for k in ("UV_CACHE_DIR", "HF_HOME") if k in os.environ
    }


def install(config, root, target=None, activate=True):
    from .api import ReadOnlyStore

    service = (
        Launchd(target, run=subprocess.run, activate=activate)
        if target
        else current().service()
    )
    env = environment() | {"SONGE_DATA_DIR": str(root)}
    identity = ReadOnlyStore(root).get("installation_id")
    if identity:
        env["SONGE_INSTALLATION_ID"] = identity
    return service.install(arguments(root), env)


def remove(target=None, activate=True, *, root=None):
    service = (
        Launchd(target, run=subprocess.run, activate=activate)
        if target
        else current().service()
    )
    if root is not None and service.target.exists():
        from .api import ReadOnlyStore

        identity = ReadOnlyStore(root).get("installation_id")
        service.check_owner(
            {"SONGE_DATA_DIR": str(root), "SONGE_INSTALLATION_ID": identity}
        )
    service.uninstall()


__all__ = ["install", "remove"]
