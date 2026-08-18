# tests/test_cli.py

import pytest

from testoai import cli


def test_cli_smoke_valid_command_prints_recommendations(artifact_path, capsys):
    cli.main(["--food-groups", "Beef Products", "--activity", "medium"])
    out = capsys.readouterr().out
    assert "Beef Products" in out


def test_cli_invalid_activity_fails_cleanly():
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--activity", "extreme"])
    assert exc_info.value.code != 0
