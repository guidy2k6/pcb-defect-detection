"""Do do ben cua mo hinh YOLO voi anh xoay, nghieng, mo, toi (ke hoach muc 2.1).

Voi moi muc nhieu trong tools/perturb.py, script dung lai tap `--split` (val hoac test) tu anh goc,
bien doi anh va nhan (box duoc bien doi theo), roi chay Ultralytics val de lay P, R, mAP50, mAP50-95.
Ket qua gop vao results/robustness.csv (moi dong khoa theo model + split + muc nhieu).

Vi du (so sanh mo hinh khong aug va co aug tren val, de chon mo hinh):
    python tools/eval_robustness.py --data configs/data_board.yaml --split val \\
        --model pcb_v8s_board=runs/detect/pcb_v8s_board/weights/best.pt \\
        --model pcb_final_aug=runs/detect/pcb_final_aug/weights/best.pt

Chay tren test chi sau khi da chot mo hinh theo val (xem huong dan trong README):
    ... --split test

Luu y khi doc ket qua:
  - R la recall trung binh cua Ultralytics (tai nguong conf cho F1 cao nhat), khong phai recall o conf co dinh.
  - Muc xoay goc le (rot15, rot45) va nghieng lam hop bao cua nhan no ra; cot label_tightness cho biet
    do khit con lai (1.0 = khit). Khi tightness thap (rot45 ~ 0.5) mAP50 bi chan tren boi chinh nhan,
    nen khong doc la "mo hinh tu".
  - Nhoe Gaussian dung sigma tinh theo pixel cua anh goc, truoc khi anh duoc thu ve imgsz.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import perturb as P  # noqa: E402
from yolo_utils import load_split, read_labels  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIELDS = ["model", "split", "perturbation", "family", "level", "n_images", "n_boxes",
          "label_tightness", "imgsz", "P", "R", "mAP50", "mAP50_95"]


def build_set(images, names, pname, root: Path, split: str, jpeg_quality: int = 95):
    """Tao tap anh + nhan da bien doi trong `root`. Tra ve (data_yaml, n_images, n_boxes, tightness)."""
    img_dir, lab_dir = root / "images" / split, root / "labels" / split
    if root.exists():
        shutil.rmtree(root)
    img_dir.mkdir(parents=True)
    lab_dir.mkdir(parents=True)
    seen, n_box, t_sum = set(), 0, 0.0
    for src in images:
        if src.stem in seen:
            raise SystemExit(f"Trung ten anh {src.stem}; ten anh phai duy nhat trong tap {split}.")
        seen.add(src.stem)
        img = cv2.imread(str(src))
        if img is None:
            raise SystemExit(f"Khong doc duoc anh {src}")
        h, w = img.shape[:2]
        cls, xyxy = P.yolo_to_xyxy(read_labels(src), w, h)
        out, nb, tight, keep = P.apply(pname, img, xyxy)
        cls = cls[keep]
        if pname == "clean":
            shutil.copyfile(src, img_dir / src.name)
        else:
            cv2.imwrite(str(img_dir / (src.stem + ".jpg")), out, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        oh, ow = out.shape[:2]
        rows = P.xyxy_to_yolo(cls, nb, ow, oh)
        (lab_dir / (src.stem + ".txt")).write_text(
            "".join(f"{c} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n" for c, x, y, bw, bh in rows), encoding="utf-8")
        n_box += len(rows)
        if len(rows) and not np.isnan(tight):
            t_sum += tight * len(rows)
    data = {"path": str(root.resolve()), "train": f"images/{split}", "val": f"images/{split}",
            "test": f"images/{split}", "names": {int(k): v for k, v in names.items()}}
    yml = root / "data.yaml"
    yml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return yml, len(images), n_box, (t_sum / n_box if n_box else float("nan"))


def ultralytics_evaluator(weights: str, split: str, imgsz: int, batch: int, device: str, work: Path):
    """Nap mo hinh mot lan; tra ve ham (data_yaml) -> dict(P, R, mAP50, mAP50_95)."""
    from ultralytics import YOLO  # import tre de test cac ham thuan khong can cai ultralytics

    model = YOLO(weights)

    def run(data_yaml: Path) -> dict:
        m = model.val(data=str(data_yaml), split=split, imgsz=imgsz, batch=batch, device=device,
                      workers=0, plots=False, verbose=False, project=str(work), name="tmp", exist_ok=True)
        return {"P": m.box.mp, "R": m.box.mr, "mAP50": m.box.map50, "mAP50_95": m.box.map}

    return run


def merge_csv(path: Path, new_rows: list[dict]):
    """Gop dong moi vao csv; dong co cung (model, split, perturbation) bi thay."""
    rows = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    key = lambda r: (r["model"], r["split"], r["perturbation"])  # noqa: E731
    fresh = {key(r) for r in new_rows}
    rows = [r for r in rows if key(r) not in fresh] + [{k: r[k] for k in FIELDS} for r in new_rows]
    order = {n: i for i, n in enumerate(P.PERTURBATIONS)}
    rows.sort(key=lambda r: (r["model"], r["split"], order.get(r["perturbation"], 99)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def parse_models(items):
    out = []
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--model phai co dang ten=duong_dan_weights, nhan duoc: {it}")
        name, w = it.split("=", 1)
        out.append((name.strip(), w.strip()))
    if len({n for n, _ in out}) != len(out):
        raise SystemExit("Ten mo hinh trong --model bi trung.")
    return out


def run(a, evaluator_factory=None):
    names, images = load_split(a.data, a.split)
    if a.limit:
        images = images[: a.limit]
    models = parse_models(a.model)
    plist = list(P.PERTURBATIONS) if a.perturbations == "all" else [x.strip() for x in a.perturbations.split(",")]
    bad = [p for p in plist if p not in P.PERTURBATIONS]
    if bad:
        raise SystemExit(f"Muc nhieu khong co: {bad}. Co: {list(P.PERTURBATIONS)}")
    cache = Path(a.cache)
    work = REPO / "runs" / "robust_eval"
    factory = evaluator_factory or (lambda w: ultralytics_evaluator(w, a.split, a.imgsz, a.batch, a.device, work))
    evaluators = {name: factory(w) for name, w in models}

    new_rows = []
    for pname in plist:
        pert = P.PERTURBATIONS[pname]
        root = cache / a.split / pname
        yml, n_img, n_box, tight = build_set(images, names, pname, root, a.split)
        print(f"[{pname}] {n_img} anh, {n_box} box, label_tightness={tight:.3f}")
        for mname, _ in models:
            m = evaluators[mname](yml)
            print(f"    {mname}: mAP50={m['mAP50']:.4f} R={m['R']:.4f} P={m['P']:.4f} mAP50-95={m['mAP50_95']:.4f}")
            new_rows.append({"model": mname, "split": a.split, "perturbation": pname, "family": pert.family,
                             "level": pert.level, "n_images": n_img, "n_boxes": n_box,
                             "label_tightness": f"{tight:.4f}", "imgsz": a.imgsz,
                             "P": f"{m['P']:.4f}", "R": f"{m['R']:.4f}",
                             "mAP50": f"{m['mAP50']:.4f}", "mAP50_95": f"{m['mAP50_95']:.4f}"})
        if not a.keep:
            shutil.rmtree(root, ignore_errors=True)
    merge_csv(Path(a.out), new_rows)
    print(f"Da ghi {a.out}")
    return new_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="yaml du lieu, vd configs/data_board.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="mac dinh val (chon mo hinh); dung test mot lan cho so cuoi")
    ap.add_argument("--model", action="append", required=True, metavar="TEN=WEIGHTS",
                    help="lap lai de so nhieu mo hinh, vd pcb_v8s_board=runs/detect/.../best.pt")
    ap.add_argument("--perturbations", default="all", help="'all' hoac danh sach cach nhau dau phay")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--cache", default=str(REPO / "dataset" / "robust"), help="thu muc tam dung tap nhieu")
    ap.add_argument("--keep", action="store_true", help="giu lai tap nhieu sau khi do (ton dia)")
    ap.add_argument("--limit", type=int, default=0, help="chi lay N anh dau (de chay thu, khong dung cho so cuoi)")
    ap.add_argument("--out", default=str(REPO / "results" / "robustness.csv"))
    a = ap.parse_args()
    if a.limit:
        print("CANH BAO: --limit dang bat; khong dung ket qua nay lam so cuoi (cot n_images ghi lai so anh).")
    run(a)


if __name__ == "__main__":
    main()
