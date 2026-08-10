from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PYPROJECT_PATH = _REPO_ROOT / "python" / "pyproject.toml"
_UV_LOCK_PATH = _REPO_ROOT / "python" / "uv.lock"


def _load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_linux_uv_source_pins_torch_to_cu126_index() -> None:
    pyproject = _load_toml(_PYPROJECT_PATH)

    extras = pyproject["project"]["optional-dependencies"]
    assert extras["train"] == ["torch==2.13.0"]
    assert "torch==2.13.0" in extras["export"]
    assert "torch==2.13.0" in extras["dev"]

    uv_config = pyproject["tool"]["uv"]
    assert uv_config["sources"]["torch"] == [
        {"index": "pytorch-cu126", "marker": "sys_platform == 'linux'"}
    ]
    assert {
        "name": "pytorch-cu126",
        "url": "https://download.pytorch.org/whl/cu126",
        "explicit": True,
    } in uv_config["index"]


def test_uv_lock_resolves_linux_torch_to_cu126_without_cu13() -> None:
    lock_text = _UV_LOCK_PATH.read_text(encoding="utf-8")
    assert "cu13" not in lock_text

    lock = tomllib.loads(lock_text)
    torch_packages = [
        package
        for package in lock["package"]
        if package["name"] == "torch" and package["version"] == "2.13.0+cu126"
    ]

    assert torch_packages
    assert torch_packages[0]["source"] == {"registry": "https://download.pytorch.org/whl/cu126"}
