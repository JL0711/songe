import hashlib
import io
import re
import subprocess
import zipfile
from pathlib import Path

import pytest
from update_formula import (
    BOOTSTRAP_DISABLE,
    POST_INSTALL,
    TARGETS,
    release_version,
    release_wheel,
    render_resources,
    resolve_resources,
    select_wheel,
    update_formula,
    verify_wheel,
)


def make_release(version="0.1.0"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("songe/cli.py", "def main(): pass\n")
        archive.writestr(
            f"songe-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: songe\nVersion: {version}\n",
        )
        archive.writestr(
            f"songe-{version}.dist-info/entry_points.txt",
            "[console_scripts]\nsonge = songe.cli:main\n",
        )
    data = buffer.getvalue()
    wheel = {
        "filename": f"songe-{version}-py3-none-any.whl",
        "url": f"https://files.pythonhosted.org/packages/songe-{version}-py3-none-any.whl",
        "digests": {"sha256": hashlib.sha256(data).hexdigest()},
        "upload_time_iso_8601": "2026-11-11T08:31:37Z",
    }
    return {"info": {"name": "songe", "version": version}, "urls": [wheel]}, data


@pytest.fixture
def release():
    return make_release()


@pytest.fixture
def resources():
    artifact = {
        "url": "https://files.pythonhosted.org/packages/installer-1.1.0-py3-none-any.whl",
        "sha256": "a" * 64,
    }
    return {target.name: {"installer": artifact} for target in TARGETS}


def test_verified_release_enables_formula_and_is_idempotent(release, resources):
    metadata, data = release
    wheel = release_wheel(metadata, "0.1.0")
    verify_wheel(data, wheel, "0.1.0")
    source = Path(__file__).with_name("songe.rb").read_text(encoding="utf-8")
    # Use a bootstrap template even after the checked-in formula is released.
    source = re.sub(r'^  version ".*"$', '  version "0.0.0.dev0"',
                    source, flags=re.MULTILINE)
    source += BOOTSTRAP_DISABLE
    updated = update_formula(source, wheel, "0.1.0", resources)
    assert 'version "0.1.0"' in updated
    assert wheel["digests"]["sha256"] in updated
    assert BOOTSTRAP_DISABLE not in updated
    assert update_formula(updated, wheel, "0.1.0", resources) == updated
    with pytest.raises(ValueError, match="downgrade"):
        update_formula(updated, wheel, "0.0.1", resources)


def test_updater_adds_noninteractive_hook_to_older_templates(release, resources):
    metadata, _ = release
    source = Path(__file__).with_name("songe.rb").read_text(encoding="utf-8")
    source = source.replace(POST_INSTALL, "")
    assert "def post_install" not in source
    updated = update_formula(source, metadata["urls"][0], "0.1.0", resources)
    assert updated.count(POST_INSTALL) == 1
    assert update_formula(updated, metadata["urls"][0], "0.1.0", resources) == updated
    assert 'quiet_system libexec/"bin/python", "-m", "songe.upgrade"' in updated


@pytest.mark.parametrize("failure", ["yanked", "host", "checksum", "project"])
def test_rejects_unusable_pypi_artifact(release, failure):
    metadata, _ = release
    wheel = metadata["urls"][0]
    if failure == "yanked":
        wheel["yanked"] = True
    elif failure == "host":
        wheel["url"] = wheel["url"].replace("files.pythonhosted.org", "example.org")
    elif failure == "checksum":
        wheel["digests"]["sha256"] = "not-a-sha256"
    else:
        metadata["info"]["name"] = "another-project"
    with pytest.raises(ValueError):
        release_wheel(metadata, "0.1.0")


def test_rejects_corrupt_wheel_and_placeholder_without_cli(release):
    metadata, data = release
    wheel = metadata["urls"][0]
    with pytest.raises(ValueError, match="SHA256"):
        verify_wheel(data + b"corrupted", wheel, "0.1.0")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "songe-0.1.0.dist-info/METADATA", "Name: songe\nVersion: 0.1.0\n"
        )
        archive.writestr("songe-0.1.0.dist-info/entry_points.txt", "")
    wheel["digests"]["sha256"] = hashlib.sha256(buffer.getvalue()).hexdigest()
    with pytest.raises(ValueError, match="CLI"):
        verify_wheel(buffer.getvalue(), wheel, "0.1.0")


@pytest.mark.parametrize("version", ["v0.1.0", "0.0.0.dev0", "0.1.0\n", "0.01.0"])
def test_rejects_invalid_versions(release, version):
    metadata, _ = release
    with pytest.raises(ValueError, match="release version"):
        release_wheel(metadata, version)


@pytest.mark.parametrize("version", ["0.1.0a2", "0.1.0b1", "0.1.0rc1", "0.1.0"])
def test_verifies_preview_and_stable_releases(version):
    metadata, data = make_release(version)
    wheel = release_wheel(metadata, version)
    verify_wheel(data, wheel, version)


def test_pep440_downgrade_order(resources):
    assert release_version("0.1.0a2") < release_version("0.1.0a10")
    assert release_version("0.1.0rc1") < release_version("0.1.0")
    metadata, _ = make_release("0.1.0a1")
    source = '  version "0.1.0a2"\n'
    with pytest.raises(ValueError, match="downgrade"):
        update_formula(source, metadata["urls"][0], "0.1.0a1", resources)


def package(name="native", version="1.0", tags=("cp312-cp312-macosx_14_0_arm64",)):
    return {"name": name, "version": version, "wheels": [
        {"url": f"https://files.pythonhosted.org/packages/{name}-{version}-{tag}.whl",
         "hashes": {"sha256": "a" * 64}}
        for tag in tags
    ]}


def test_selects_one_native_wheel_and_rejects_newer_os_or_wrong_arch():
    p = package(tags=("cp312-cp312-macosx_26_0_arm64",
                      "cp312-cp312-macosx_14_0_arm64", "py3-none-any"))
    assert select_wheel(p, TARGETS[0])["url"].endswith("macosx_14_0_arm64.whl")
    assert select_wheel(p, TARGETS[1])["url"].endswith("py3-none-any.whl")
    with pytest.raises(ValueError, match="No compatible wheel"):
        select_wheel(package(), TARGETS[1])


@pytest.mark.parametrize("target", TARGETS[2:])
def test_linux_native_wheels(target):
    arch = "aarch64" if target.arch == "arm" else "x86_64"
    p = package(tags=(f"cp312-cp312-manylinux_2_28_{arch}",
                      f"cp312-cp312-manylinux_2_35_{arch}"))
    assert select_wheel(p, target)["url"].endswith(f"manylinux_2_28_{arch}.whl")


@pytest.mark.parametrize("failure", ["hash", "host", "name", "version", "sdist"])
def test_rejects_bad_dependency_artifacts(failure):
    p = package()
    if failure == "hash":
        p["wheels"][0]["hashes"]["sha256"] = "bad"
    elif failure == "host":
        p["wheels"][0]["url"] = p["wheels"][0]["url"].replace(
            "files.pythonhosted.org", "example.com")
    elif failure == "name":
        p["name"] = "other"
    elif failure == "version":
        p["version"] = "2.0"
    else:
        p.pop("wheels")
    with pytest.raises(ValueError):
        select_wheel(p, TARGETS[0])


def test_resources_share_universal_wheels_and_separate_native_wheels(resources):
    for target in TARGETS:
        resources[target.name]["native"] = select_wheel(
            package(tags=("py3-none-any",)), target)
    # Only Apple Silicon has the MLX dependency.
    resources[TARGETS[0].name]["mlx"] = select_wheel(
        package(name="mlx"), TARGETS[0])
    rendered = render_resources(resources)
    assert rendered.count('resource "installer"') == 1
    assert rendered.count('resource "native"') == 1
    assert rendered.count('resource "mlx"') == 1
    assert rendered.index('on_macos') < rendered.index('resource "mlx"')
    assert rendered.index('resource "mlx"') < rendered.index('on_linux')
    resources.pop(TARGETS[0].name)
    with pytest.raises(ValueError, match="complete wheel set"):
        render_resources(resources)


def test_resolver_uses_published_wheel_and_explicit_platforms(monkeypatch, release):
    metadata, _ = release
    wheel = metadata["urls"][0]
    commands = []
    monkeypatch.setenv("UV_INDEX", "https://example.com/unexpected")
    monkeypatch.setenv("UV_CACHE_DIR", "/tmp/songe-test-cache")

    def run(command, *, env, check, timeout):
        assert check and timeout == 600
        assert "UV_INDEX" not in env
        assert env["UV_CACHE_DIR"] == "/tmp/songe-test-cache"
        assert env["MACOSX_DEPLOYMENT_TARGET"] == "14.0"
        requirements = Path(command[5]).read_text(encoding="utf-8")
        assert wheel["url"] + "#sha256=" + wheel["digests"]["sha256"] in requirements
        assert "installer==1.1.0" in requirements
        output = Path(command[command.index("--output-file") + 1])
        output.write_text(
            '[[packages]]\nname="songe"\nversion="0.1.0"\n'
            '[[packages]]\nname="installer"\nversion="1.1.0"\n'
            'wheels=[{url="https://files.pythonhosted.org/installer-1.1.0-py3-none-any.whl",'
            'hashes={sha256="' + "a" * 64 + '"}}]\n', encoding="utf-8")
        commands.append(command)

    monkeypatch.setattr(subprocess, "run", run)
    result = resolve_resources(wheel, "0.1.0")
    assert set(result) == {target.name for target in TARGETS}
    assert [command[command.index("--python-platform") + 1] for command in commands] == [
        target.uv_platform for target in TARGETS]
    assert all("--only-binary" in command and "--exclude-newer" in command
               and "--no-config" in command for command in commands)
