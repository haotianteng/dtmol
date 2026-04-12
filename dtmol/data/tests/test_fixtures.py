"""Smoke tests that conftest fixtures are loadable."""


def test_data_root_is_path(data_root):
    from pathlib import Path

    assert isinstance(data_root, Path)


def test_unified_config_fields(unified_config):
    assert unified_config.max_seq_len == 128
    assert unified_config.max_pocket_atoms == 64
