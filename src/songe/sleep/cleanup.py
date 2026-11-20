"""Collect terminal work owned by this installation; retain resumable units."""

import json
import shutil
from pathlib import Path

from ..artifacts.runtime import owned_marker, remove_owned
from ..config import training_fingerprint


def collect_work(store, config, protected=()):
    # An interrupted unit can only resume under its original binding/config.
    from ..schedule import base_fingerprint, resumable

    configuration, base = training_fingerprint(config), base_fingerprint(config)
    for night in store.nights():
        if night["status"] == "interrupted" and not resumable(
            night, store, config, configuration=configuration, base=base
        ):
            store.night(night["id"], status="discarded")
    retained = set()
    if config.model:
        retained.add(Path(config.model).resolve())
    with store.connect() as db:
        for row in db.execute("SELECT artifact,adapter,training_adapter FROM versions"):
            retained.update(Path(p).resolve() for p in row if p)
    accepted = {n["id"] for n in store.nights() if n["status"] == "accepted"}
    terminal = {
        n["id"]
        for n in store.nights()
        if n["status"] in {"failed", "rejected", "discarded", "waiting"}
    }
    identity = store.get("installation_id")
    from ..artifacts.storage import artifact_roots

    for marker in (
        marker
        for directory in artifact_roots(store.root)
        for marker in (directory / "versions").glob("night-*/songe_work.json")
    ):
        try:
            meta = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        work = marker.parent.resolve()
        if marker.parent.is_symlink():
            continue
        if (
            meta.get("installation_id") != identity
            or meta.get("night") not in terminal | accepted
            or work.name != f"night-{meta['night']}"
        ):
            continue
        if any(p == work or p.is_relative_to(work) for p in retained):
            if meta.get("night") in accepted:
                # Keep every published adapter, and only the current export.
                # Drop optimizer checkpoints, unused scales and their
                # evaluation conversions from this owned, terminal work tree.
                live = {store.get("current")} | set(protected)
                from ..models.registry import entries

                for model in entries(store):
                    scoped = store.scoped(model["id"])
                    live.add(scoped.get("current"))
                with store.connect() as db:
                    versions = [dict(v) for v in db.execute("SELECT * FROM versions")]
                keep = {
                    Path(v[k]).resolve()
                    for v in versions
                    for k in ("adapter", "training_adapter")
                    if v.get(k)
                }
                keep.update(
                    Path(v["artifact"]).resolve()
                    for v in versions
                    if v["id"] in live or not v.get("adapter")
                )
                if config.model:
                    keep.add(Path(config.model).resolve())
                for child in marker.parent.iterdir():
                    if child.name == "songe_work.json" or any(
                        p == child.resolve() or p.is_relative_to(child.resolve())
                        for p in keep
                    ):
                        continue
                    collect_derived(store.root, child, [], keep)
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                # Other evaluated scale exports can live in the library.
                if config.models_dir:
                    for candidate in (Path(config.models_dir) / config.publisher).rglob(
                        "*songe_owned.json"
                    ):
                        try:
                            data = json.loads(candidate.read_text(encoding="utf-8"))
                            artifact = (
                                candidate.parent
                                if candidate.name == "songe_owned.json"
                                else candidate.with_name(
                                    candidate.name.removesuffix(".songe_owned.json")
                                )
                            )
                            if (
                                data.get("installation_id") == identity
                                and Path(data.get("adapter", "/")).is_relative_to(work)
                                and artifact.resolve() not in keep
                            ):
                                collect_derived(store.root, work, [artifact], keep)
                                remove_owned(artifact, identity)
                        except (OSError, ValueError):
                            continue
            continue
        # Fusions can live in the LM Studio library rather than the data root.
        artifacts = []
        if config.models_dir:
            for artifact_marker in (Path(config.models_dir) / config.publisher).rglob(
                "*songe_owned.json"
            ):
                try:
                    data = json.loads(artifact_marker.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                adapter = Path(data.get("adapter", "/"))
                if data.get(
                    "installation_id"
                ) != identity or not adapter.is_relative_to(work):
                    continue
                artifact = (
                    artifact_marker.parent
                    if artifact_marker.name == "songe_owned.json"
                    else artifact_marker.with_name(
                        artifact_marker.name.removesuffix(".songe_owned.json")
                    )
                )
                if (
                    artifact.resolve() not in retained
                    and owned_marker(artifact).is_file()
                ):
                    artifacts.append(artifact)
        collect_derived(store.root, work, artifacts, retained)
        for artifact in artifacts:
            remove_owned(artifact, identity)
        shutil.rmtree(work)


def collect_derived(root, work, artifacts, retained):
    sources = {}
    from ..artifacts.storage import artifact_roots

    for marker in (
        marker
        for directory in artifact_roots(root)
        for marker in (directory / "models").glob("*/songe_conversion.json")
    ):
        try:
            if marker.parent.is_symlink():
                continue
            sources[marker.parent.resolve()] = Path(
                json.loads(marker.read_text(encoding="utf-8"))["source"]
            ).resolve()
        except (OSError, ValueError, KeyError):
            continue
    doomed = set()
    owned = {p.resolve() for p in artifacts}
    while True:
        added = {
            cache
            for cache, source in sources.items()
            if cache not in doomed
            and not any(p == cache or p.is_relative_to(cache) for p in retained)
            and (source in owned or source in doomed or source.is_relative_to(work))
        }
        if not added:
            break
        doomed.update(added)
    for cache in doomed:
        shutil.rmtree(cache)
