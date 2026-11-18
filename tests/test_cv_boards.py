import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import cv_boards as C  # noqa: E402

BOARDS = ["01", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
NAMES = ["Missing Hole", "Mouse Bite", "Open Circuit", "Short", "Spur", "Spurious Copper"]


def make_tree(root):
    """Cach chia chinh: train 6 bo, val 2, test 2; anh nam trong thu muc con theo lop nhu divide_data.py."""
    split_of = {b: "train" for b in BOARDS}
    split_of.update({"07": "val", "11": "val", "08": "test", "12": "test"})
    for b in BOARDS:
        s = split_of[b]
        img_dir, lab_dir = root / "images" / s / "Short", root / "labels" / s / "Short"
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            cv2.imwrite(str(img_dir / f"{b}_x_{i}.jpg"), np.zeros((20, 20, 3), np.uint8))
            (lab_dir / f"{b}_x_{i}.txt").write_text("".join(f"{c} 0.5 0.5 0.1 0.1\n" for c in range(6)))
    return root


def test_folds_partition_test_boards_and_never_leak():
    folds = C.make_folds(BOARDS)
    assert len(folds) == 5
    tests = [b for f in folds for b in f["test"]]
    assert sorted(tests) == sorted(BOARDS) and len(set(tests)) == 10   # moi bo lam test dung mot lan
    for f in folds:
        assert len(f["test"]) == len(f["val"]) == 2 and len(f["train"]) == 6
        assert not set(f["test"]) & set(f["val"]) and not set(f["test"]) & set(f["train"]) and not set(f["val"]) & set(f["train"])
    with pytest.raises(SystemExit):
        C.make_folds(BOARDS[:9])


def test_list_images_groups_by_board_across_original_splits(tmp_path):
    by = C.list_images(make_tree(tmp_path))
    assert list(by) == sorted(BOARDS) and all(len(v) == 2 for v in by.values())


def test_write_fold_files_point_at_the_right_boards(tmp_path):
    by = C.list_images(make_tree(tmp_path))
    folds = C.make_folds(list(by))
    yml = C.write_fold(0, folds[0], by, dict(enumerate(NAMES)), tmp_path / "cv")
    cfg = yaml.safe_load(yml.read_text())
    assert cfg["names"][5] == "Spurious Copper" and cfg["test"] == "test.txt"
    d = yml.parent
    test_ids = {Path(l).stem.split("_")[0] for l in (d / "test.txt").read_text().split()}
    train_ids = {Path(l).stem.split("_")[0] for l in (d / "train.txt").read_text().split()}
    assert test_ids == set(folds[0]["test"]) and not test_ids & train_ids
    assert len((d / "train.txt").read_text().split()) == 12


def test_summarize_mean_std_and_partial_label():
    rows = [{"P": 0.8, "R": 0.9, "mAP50": 0.9, "mAP50_95": 0.5}, {"P": 0.6, "R": 0.7, "mAP50": 0.7, "mAP50_95": 0.3}]
    m, s = C.summarize(rows)
    assert m["fold"] == "mean(n=2)" and m["mAP50"] == pytest.approx(0.8)
    assert s["mAP50"] == pytest.approx(0.1414213, abs=1e-6)   # do lech chuan mau, ddof=1


def test_dry_run_cli_writes_lists_and_trains_nothing(tmp_path):
    make_tree(tmp_path / "ds")
    (tmp_path / "d.yaml").write_text(yaml.safe_dump({"names": dict(enumerate(NAMES))}))
    r = subprocess.run([sys.executable, str(REPO / "tools" / "cv_boards.py"), "--dataset", str(tmp_path / "ds"),
                        "--data", str(tmp_path / "d.yaml"), "--cv-root", str(tmp_path / "cv"), "--dry-run"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("fold") >= 5 and "CANH BAO" not in r.stdout
    assert (tmp_path / "cv" / "fold4" / "data.yaml").exists() and not (tmp_path / "runs").exists()
