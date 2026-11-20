"""The pipeline source is the authority for the backend surface."""

import ast
import inspect
from pathlib import Path

import pytest

from songe.backends.mlx import Backend as MLXBackend
from songe.backends.torch import TorchBackend

PIPELINE = Path(__file__).parents[1] / "src/songe/sleep"


def attributes(cls):
    tree = ast.parse(inspect.getsource(cls))
    names = {
        n.name
        for n in tree.body[0].body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            names.add(node.attr)
    names.update(vars(cls))
    return names


def pipeline_attributes():
    names = set()
    for name in ("worker", "select", "evaluate", "calibrate", "probe", "fuse"):
        tree = ast.parse((PIPELINE / f"{name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in {"backend", "candidate", "base", "incumbent"}
            ):
                names.add(node.attr)
    return names


@pytest.mark.parametrize("backend", [MLXBackend, TorchBackend])
def test_backend_implements_the_surface_used_by_the_pipeline(backend):
    required = pipeline_attributes()
    assert required  # Never silently pass an empty extraction.
    assert not required - attributes(backend)
