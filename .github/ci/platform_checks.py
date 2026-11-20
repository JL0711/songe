"""Report platform gaps without importing unavailable APIs or GPU runtimes.

This is test infrastructure only. Remove the Windows exceptions as the platform
workstream replaces POSIX locks and permission assertions in the existing tests.
"""

import importlib
import importlib.metadata
import platform
import sys
from functools import cache

import pytest


@cache
def metal_unavailable():
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return "Metal tests require an Apple Silicon Mac"
    try:
        mx = importlib.import_module("mlx.core")
        if not mx.metal.is_available():
            return "Metal is unavailable in this session"
    except (ImportError, RuntimeError) as error:
        return f"Metal is unavailable: {error}"
    return None


def module_unavailable(name):
    if name == "test_compat.py":
        return metal_unavailable()
    return None


class UnavailableModule(pytest.Module):
    def collect(self):
        pytest.skip(module_unavailable(self.path.name), allow_module_level=True)


def pytest_pycollect_makemodule(module_path, parent):
    if module_unavailable(module_path.name):
        return UnavailableModule.from_parent(parent, path=module_path)
    return None


def pytest_collection_modifyitems(items):
    try:
        importlib.metadata.version("mlx-lm")
        has_mlx_lm = True
    except importlib.metadata.PackageNotFoundError:
        has_mlx_lm = False
    for item in items:
        if (
            not has_mlx_lm
            and item.name
            == "test_unsupported_and_known_slow_architectures_are_excluded_without_metal"
        ):
            item.add_marker(
                pytest.mark.skip(reason="Requires installed MLX-LM metadata")
            )
        if (
            sys.platform == "win32"
            and item.name == "test_durable_atomic_swap_and_restart"
        ):
            item.add_marker(
                pytest.mark.skip(reason="Existing test asserts POSIX mode 0700")
            )
