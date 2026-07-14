"""
Unit tests for chain.toml resolution.

chain.toml is consensus-critical and is read at IMPORT time, so it must ship
inside the package (leoma/chain.toml) — not merely exist in the git checkout.
It previously lived at the repo root and was resolved via ``parents[2]``, which
meant a non-editable install resolved it into ``site-packages/`` and crashed with
FileNotFoundError on startup. These tests pin the resolution contract.
"""

import pathlib

import pytest

from leoma.infra import chain_config
from leoma.infra.chain_config import _resolve_toml_path


class TestPackagedLocation:
    def test_chain_toml_lives_inside_the_package(self):
        """The file must sit next to the package, so package-data ships it."""
        pkg_dir = pathlib.Path(chain_config.__file__).resolve().parents[1]  # .../leoma/
        assert (pkg_dir / "chain.toml").is_file(), (
            "chain.toml must live at leoma/chain.toml so it ships in the wheel"
        )

    def test_resolved_path_is_the_packaged_one(self):
        resolved = _resolve_toml_path()
        pkg_dir = pathlib.Path(chain_config.__file__).resolve().parents[1]
        assert resolved.is_file()
        assert resolved.resolve() == (pkg_dir / "chain.toml").resolve()

    def test_config_actually_loaded(self):
        assert chain_config.NAME == "leoma"
        assert chain_config.ARCH_PIPELINE  # the one live [arch] key


class TestOverride:
    def test_override_absolute_path(self, tmp_path, monkeypatch):
        alt = tmp_path / "alt.toml"
        alt.write_text('[chain]\nname = "alt"\nseed_repo = "x/y"\n')
        monkeypatch.setenv("LEOMA_CHAIN_OVERRIDE", str(alt))
        assert _resolve_toml_path() == alt

    def test_override_relative_to_cwd(self, tmp_path, monkeypatch):
        alt = tmp_path / "rel.toml"
        alt.write_text('[chain]\nname = "rel"\nseed_repo = "x/y"\n')
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("LEOMA_CHAIN_OVERRIDE", "rel.toml")
        assert _resolve_toml_path().resolve() == alt.resolve()

    def test_missing_override_raises_clearly(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LEOMA_CHAIN_OVERRIDE", str(tmp_path / "nope.toml"))
        with pytest.raises(RuntimeError, match="missing file"):
            _resolve_toml_path()
