import csv
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import eval_robustness as E  # noqa: E402
import perturb as P  # noqa: E402

NAMES = ["Missing Hole", "Mouse Bite", "Open Circuit", "Short", "Spur", "Spurious Copper"]


def make_dataset(tmp_path, n=3, w=640, h=480):
    img_dir, lab_dir = tmp_path / "images" / "val", tmp_path / "labels" / "val"
    img_dir.mkdir(parents=True), lab_dir.mkdir(parents=True)
    for i in range(n):
        img = np.zeros((h, w, 3), np.uint8)
        x1, y1, x2, y2 = 200, 150, 240, 190
        img[y1:y2, x1:x2] = 255
        cv2.imwrite(str(img_dir / f"07_a_{i}.jpg"), img)
        (lab_dir / f"07_a_{i}.txt").write_text(f"3 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {40 / w:.6f} {40 / h:.6f}\n")
    cfg = {"path": str(tmp_path), "train": "images/val", "val": "images/val", "names": dict(enumerate(NAMES))}
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(cfg))
    return tmp_path / "data.yaml"


def args(tmp_path, data, **kw):
    base = dict(data=str(data), split="val", model=["old=w1.pt", "new=w2.pt"], perturbations="all", imgsz=1024,
                batch=8, device="cpu", cache=str(tmp_path / "cache"), keep=False, limit=0, out=str(tmp_path / "rob.csv"))
    base.update(kw)
    return Namespace(**base)


def fake_factory(weights):
    def run(yml):
        return {"P": 0.9, "R": 0.8, "mAP50": 0.7 if weights == "w1.pt" else 0.75, "mAP50_95": 0.4}
    return run


def test_run_writes_all_levels_for_both_models(tmp_path):
    data = make_dataset(tmp_path)
    E.run(args(tmp_path, data), evaluator_factory=fake_factory)
    rows = list(csv.DictReader(open(tmp_path / "rob.csv", encoding="utf-8")))
    assert len(rows) == 2 * len(P.PERTURBATIONS)
    assert {r["model"] for r in rows} == {"old", "new"}
    assert [r["perturbation"] for r in rows if r["model"] == "old"] == list(P.PERTURBATIONS)
    assert list(rows[0].keys()) == E.FIELDS
    assert not (tmp_path / "cache" / "val").exists() or not any((tmp_path / "cache" / "val").iterdir())  # da don tap tam


def test_built_set_has_transformed_labels_that_follow_the_content(tmp_path):
    data = make_dataset(tmp_path)
    names, images = E.load_split(str(data), "val")
    for pname in ["clean", "rot90", "rot180", "rot15", "tilt10", "blur_s3", "dark50"]:
        root = tmp_path / "set" / pname
        yml, n_img, n_box, tight = E.build_set(images, names, pname, root, "val")
        assert n_img == 3 and n_box == 3
        cfg = yaml.safe_load(yml.read_text())
        assert cfg["names"][3] == "Short" and cfg["test"] == "images/val"
        out_img = cv2.imread(str(next((root / "images" / "val").glob("*.jpg"))))
        h, w = out_img.shape[:2]
        lab = next((root / "labels" / "val").glob("*.txt")).read_text().split()
        c, xc, yc, bw, bh = int(lab[0]), *map(float, lab[1:])
        assert c == 3
        if pname in ("blur_s3", "dark50"):
            continue  # noi dung bi lam mo/toi, khong co ranh gioi cung de doi chieu
        ys, xs = np.where(out_img[..., 0] > 128)
        cx, cy = (xs.min() + xs.max() + 1) / 2 / w, (ys.min() + ys.max() + 1) / 2 / h
        assert abs(xc - cx) < 0.01 and abs(yc - cy) < 0.01, pname  # tam box trung tam noi dung
        if pname in ("clean", "rot90", "rot180"):
            assert abs(bw * w - (xs.max() + 1 - xs.min())) < 3 and abs(bh * h - (ys.max() + 1 - ys.min())) < 3
        if pname == "clean":
            assert tight == pytest.approx(1.0)
        if pname == "rot15":
            assert tight < 1.0


def test_rot90_swaps_image_dimensions(tmp_path):
    data = make_dataset(tmp_path, w=640, h=480)
    names, images = E.load_split(str(data), "val")
    E.build_set(images, names, "rot90", tmp_path / "r", "val")
    h, w = cv2.imread(str(next((tmp_path / "r" / "images" / "val").glob("*.jpg")))).shape[:2]
    assert (w, h) == (480, 640)


def test_merge_replaces_same_key_and_keeps_other_rows(tmp_path):
    data = make_dataset(tmp_path)
    out = tmp_path / "rob.csv"
    E.run(args(tmp_path, data, perturbations="clean,rot90"), evaluator_factory=fake_factory)
    E.run(args(tmp_path, data, perturbations="rot90,blur_s1", model=["old=w1.pt"]), evaluator_factory=fake_factory)
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    keys = [(r["model"], r["perturbation"]) for r in rows]
    assert len(keys) == len(set(keys)) == 5  # old: clean, rot90, blur_s1; new: clean, rot90
    assert ("new", "rot90") in keys and ("old", "blur_s1") in keys


def test_limit_and_bad_inputs(tmp_path):
    data = make_dataset(tmp_path)
    rows = E.run(args(tmp_path, data, perturbations="clean", limit=2), evaluator_factory=fake_factory)
    assert rows[0]["n_images"] == 2
    with pytest.raises(SystemExit):
        E.run(args(tmp_path, data, perturbations="rot33"), evaluator_factory=fake_factory)
    with pytest.raises(SystemExit):
        E.parse_models(["no_equals_sign"])
    with pytest.raises(SystemExit):
        E.parse_models(["a=w1.pt", "a=w2.pt"])
