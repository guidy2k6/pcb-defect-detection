import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import train as T  # noqa: E402  (train.py chi import ultralytics luc chay that)


def test_robust_config_loads_and_matches_plan():
    kw, spec = T.load_aug(REPO / "configs" / "aug_robust.yaml")
    assert kw["flipud"] == 0.5 and kw["fliplr"] == 0.5
    assert 0 < kw["degrees"] <= 15          # goc nho, khong qua 15 do theo ke hoach
    assert 0 < kw["perspective"] <= 0.001   # khoang Ultralytics khuyen nghi
    names = [s["name"] for s in spec]
    assert {"RandomRotate90", "Blur", "MedianBlur", "MotionBlur"} <= set(names)
    assert all(0 < s["p"] <= 1 for s in spec)


def test_unknown_keys_fail_early(tmp_path):
    f = tmp_path / "a.yaml"
    f.write_text("ultralytics:\n  degreez: 5\n")
    with pytest.raises(SystemExit):
        T.load_aug(f)
    f.write_text("tensorflow: {}\n")
    with pytest.raises(SystemExit):
        T.load_aug(f)
    f.write_text("albumentations:\n  - {p: 0.5}\n")
    with pytest.raises(SystemExit):
        T.load_aug(f)


def test_albumentations_specs_build_when_library_present():
    A = pytest.importorskip("albumentations")
    _, spec = T.load_aug(REPO / "configs" / "aug_robust.yaml")
    built = T.build_albumentations(spec)
    assert len(built) == len(spec) and isinstance(built[0], A.RandomRotate90)
    with pytest.raises(SystemExit):
        T.build_albumentations([{"name": "NoSuchTransform"}])


def test_log_row_has_aug_column(tmp_path, monkeypatch):
    import types
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(__version__="9.9.9"))
    monkeypatch.setattr(T, "LOG", tmp_path / "train_params.md")
    a = types.SimpleNamespace(name="r", model="m.pt", data="d.yaml", imgsz=1024, batch=8, epochs=1, patience=1, seed=0,
                              aug="configs/aug_robust.yaml")
    T.log_run(a)
    a.aug, a.name = None, "r2"
    T.log_run(a)
    lines = (tmp_path / "train_params.md").read_text(encoding="utf-8").splitlines()
    assert lines[2].endswith("| aug |") and lines[-2].endswith("| configs/aug_robust.yaml |") and lines[-1].endswith("| mac dinh |")
    assert len({l.count("|") for l in lines if l.startswith("|")}) == 1  # moi dong cung so cot
