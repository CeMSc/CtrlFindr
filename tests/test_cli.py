from ctrlfindr.__main__ import main
from fwutil import make_framework


def test_template_command(tmp_path, capsys):
    assert main(["template", str(tmp_path / "f.xlsx")]) == 0
    assert (tmp_path / "f.xlsx").is_file()
    assert main(["template", str(tmp_path / "f.xlsx")]) == 2  # never overwrites
    assert main(["template", str(tmp_path / "f.csv")]) == 2


def test_check_command(tmp_path):
    (tmp_path / "Input").mkdir()
    fw = make_framework(tmp_path / "Input" / "fw.ods", [("haz", "flood*")], [("h", "haz")], [("1", "Hazard", "h")],
                        tests=[("Floods came.", "h", "")])
    (tmp_path / "ctrlfindr.toml").write_text(f'[paths]\nframework = "{fw.as_posix()}"\n', encoding="utf-8")
    assert main(["check", "-c", str(tmp_path / "ctrlfindr.toml")]) == 0
    bad = make_framework(tmp_path / "Input" / "bad.ods", [("haz", "flood*")], [("h", "missing")], [("1", "H", "h")])
    (tmp_path / "bad.toml").write_text(f'[paths]\nframework = "{bad.as_posix()}"\n', encoding="utf-8")
    assert main(["check", "-c", str(tmp_path / "bad.toml")]) == 2
    assert main(["check", "-c", str(tmp_path / "nope.toml")]) == 2


def test_stop_returns_130(tmp_path, monkeypatch, capsys):
    from ctrlfindr import ingest

    (tmp_path / "ctrlfindr.toml").write_text("", encoding="utf-8")
    monkeypatch.setattr(ingest, "run", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt))
    assert main(["ingest", "-c", str(tmp_path / "ctrlfindr.toml"), "--no-progress"]) == 130
    assert "run the same command again" in capsys.readouterr().err.lower()
