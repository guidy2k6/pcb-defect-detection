import csv
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import eval_robustness as E  # noqa: E402
import plot_robustness as R  # noqa: E402


def fake_csv(path):
    rows = []
    for m, base in (("old", 0.9), ("new", 0.95)):
        for i, (n, fam, lvl) in enumerate([("clean", "clean", 0), ("rot15", "rot", 15), ("rot45", "rot", 45), ("rot90", "rot", 90),
                                           ("tilt5", "tilt", 0.05), ("tilt10", "tilt", 0.10), ("blur_s1", "blur", 1),
                                           ("blur_s3", "blur", 3), ("motion15", "motion", 15), ("dark50", "dark", 0.5)]):
            v = max(base - 0.03 * i, 0.1)
            rows.append({"model": m, "split": "val", "perturbation": n, "family": fam, "level": lvl, "n_images": 1, "n_boxes": 1,
                         "label_tightness": "0.5000" if n == "rot45" else "1.0000", "imgsz": 1024,
                         "P": f"{v:.4f}", "R": f"{v - 0.02:.4f}", "mAP50": f"{v:.4f}", "mAP50_95": f"{v / 2:.4f}"})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=E.FIELDS)
        w.writeheader()
        w.writerows(rows)


def test_table_values_come_from_csv(tmp_path):
    fake_csv(tmp_path / "r.csv")
    rows = R.load(tmp_path / "r.csv", "val")
    R.make_table(rows, tmp_path / "t.tex", "val")
    t = (tmp_path / "t.tex").read_text(encoding="utf-8")
    assert "Xoay 45°" in t and "0.50" in t          # độ khít nhãn hiện trong bảng
    assert "0.9500" in t and "0.9000" in t           # mAP50 của hai mô hình ở mức gốc
    assert t.count(r"\\") >= 12 and r"\begin{tabular}" in t
    assert "pcb_" not in t or r"\_" in t             # gạch dưới đã escape


def test_figure_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    fake_csv(tmp_path / "r.csv")
    R.make_figure(R.load(tmp_path / "r.csv", "val"), tmp_path / "f.png", "val")
    assert (tmp_path / "f.png").stat().st_size > 10_000


def test_missing_split_is_an_error(tmp_path):
    fake_csv(tmp_path / "r.csv")
    with pytest.raises(SystemExit):
        R.load(tmp_path / "r.csv", "test")
