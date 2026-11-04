"""Configuration relocation checks; no provider, runtime or device access."""
from PhyAgentOS.config import loader

def test_export_config_path(monkeypatch, tmp_path):
    path = tmp_path / "instance/config.json"
    monkeypatch.setattr(loader, "_current_config_path", None)
    monkeypatch.setenv("PAOS_CONFIG", str(path))
    assert loader.get_config_path() == path

def test_explicit_config_context_wins(monkeypatch, tmp_path):
    selected = tmp_path / "explicit.json"
    monkeypatch.setattr(loader, "_current_config_path", selected)
    monkeypatch.setenv("PAOS_CONFIG", str(tmp_path / "default.json"))
    assert loader.get_config_path() == selected
