"""Kiem dinh cheo theo bo mach: 5 fold x 2 bo (ke hoach muc 2.3, viec tuy chon).

Moi fold: 2 bo lam test (giu ra ngoai, chi do mot lan), 2 bo ke tiep lam val (early stopping, chon epoch),
6 bo con lai train. Moi bo lam test dung mot lan. Bao cao trung binh +- do lech chuan (mau, ddof=1) giua 5 fold.

    python tools/cv_boards.py --dataset dataset/pcb_boards_yolo --model yolov8s.pt \\
        --aug configs/aug_robust.yaml --device 0                 # train + do ca 5 fold
    python tools/cv_boards.py --dataset ... --folds 0,1 --dry-run   # chi in cach chia, khong train
    python tools/cv_boards.py --dataset ... --eval-only             # do lai tu cac run da co (vd train tren Colab)

Chi dung de bao cao do on dinh cua cach huan luyen, khong dung de chon lai mo hinh cuoi.
Luu y: bo 07, 11 (val) va 08, 12 (test) cua cach chia chinh co the nam o fold nao do lam test;
day khong phai ro ri vi moi fold huan luyen lai tu dau.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yolo_utils import IMG_EXT, read_labels  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
N_FOLDS = 5


def list_images(dataset: Path):
    """Moi anh trong images/{train,val,test}, nhom theo ID bo (phan truoc dau '_' dau tien cua ten anh)."""
    boards: dict[str, list[Path]] = {}
    for split in ("train", "val", "test"):
        d = dataset / "images" / split
        if not d.is_dir():
            raise SystemExit(f"Khong thay {d}. Chay data/io/divide_data.py truoc.")
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in IMG_EXT:
                boards.setdefault(p.stem.split("_")[0], []).append(p.resolve())
    return {k: sorted(v) for k, v in sorted(boards.items())}


def make_folds(board_ids: list[str], n_folds: int = N_FOLDS):
    """Tra ve list fold {test, val, train} (moi gia tri la list ID bo). Test cua cac fold phan hoach toan bo bo."""
    b = sorted(board_ids)
    if len(b) % n_folds:
        raise SystemExit(f"{len(b)} bo khong chia deu cho {n_folds} fold.")
    k = len(b) // n_folds
    chunks = [b[i * k:(i + 1) * k] for i in range(n_folds)]
    folds = []
    for i in range(n_folds):
        test, val = chunks[i], chunks[(i + 1) % n_folds]
        folds.append({"test": test, "val": val, "train": [x for x in b if x not in test and x not in val]})
    return folds


def classes_present(images):
    s = set()
    for im in images:
        s.update(c for c, *_ in read_labels(im))
    return s


def write_fold(fold_idx: int, fold: dict, by_board: dict, names: dict, out_root: Path):
    d = out_root / f"fold{fold_idx}"
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"path": str(d.resolve()), "names": {int(k): v for k, v in names.items()}}
    for split in ("train", "val", "test"):
        imgs = [p for b in fold[split] for p in by_board[b]]
        (d / f"{split}.txt").write_text("".join(f"{p}\n" for p in imgs), encoding="utf-8")
        cfg[split] = f"{split}.txt"
    (d / "data.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return d / "data.yaml"


def summarize(rows: list[dict]):
    """rows: dict(fold, P, R, mAP50, mAP50_95 dang so). Tra ve dong mean va std (mau, ddof=1).
    Nhan ghi ro so fold, vd 'mean(n=5)', de mot lan chay chi vai fold khong bi doc nham la du 5 fold."""
    out = []
    for stat, fn in (("mean", statistics.mean), ("std", lambda v: statistics.stdev(v) if len(v) > 1 else float("nan"))):
        out.append({"fold": f"{stat}(n={len(rows)})",
                    **{k: fn([float(r[k]) for r in rows]) for k in ("P", "R", "mAP50", "mAP50_95")}})
    return out


def evaluate(weights: Path, data_yaml: Path, imgsz: int, batch: int, device: str):
    from ultralytics import YOLO

    m = YOLO(str(weights)).val(data=str(data_yaml), split="test", imgsz=imgsz, batch=batch, device=device,
                               workers=0, plots=False, verbose=False,
                               project=str(REPO / "runs" / "cv_eval"), name=weights.parents[1].name, exist_ok=True)
    return {"P": m.box.mp, "R": m.box.mr, "mAP50": m.box.map50, "mAP50_95": m.box.map}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="thu muc co images/{train,val,test} va labels/... (divide_data.py)")
    ap.add_argument("--data", default=str(REPO / "configs" / "data_board.yaml"), help="lay ten lop")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--aug", default=None, help="vd configs/aug_robust.yaml (bo trong = mac dinh)")
    ap.add_argument("--prefix", default="pcb_cv", help="ten run: <prefix>_fold<k>")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--device", default="0")
    ap.add_argument("--dry-run", action="store_true", help="chi in cach chia fold va ghi file danh sach anh")
    ap.add_argument("--eval-only", action="store_true", help="khong train, do lai tu runs/detect/<prefix>_fold<k>")
    ap.add_argument("--cv-root", default=str(REPO / "dataset" / "cv"), help="noi ghi danh sach anh va yaml tung fold")
    ap.add_argument("--out", default=str(REPO / "results" / "cv_boards.csv"))
    a = ap.parse_args()

    names = yaml.safe_load(Path(a.data).read_text(encoding="utf-8"))["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    by_board = list_images(Path(a.dataset))
    folds = make_folds(list(by_board))
    chosen = [int(x) for x in a.folds.split(",")]
    if any(i < 0 or i >= len(folds) for i in chosen):
        raise SystemExit(f"--folds phai nam trong 0..{len(folds) - 1}")

    rows = []
    for i in chosen:
        f = folds[i]
        yml = write_fold(i, f, by_board, names, Path(a.cv_root))
        n = {s: sum(len(by_board[b]) for b in f[s]) for s in ("train", "val", "test")}
        miss = {s: sorted(set(names) - classes_present([p for b in f[s] for p in by_board[b]])) for s in ("val", "test")}
        print(f"fold{i}: test={f['test']} val={f['val']} train={f['train']}  anh {n}")
        for s, m in miss.items():
            if m:
                print(f"  CANH BAO: tap {s} cua fold{i} khong co lop {[names[c] for c in m]}")
        if a.dry_run:
            continue
        run = f"{a.prefix}_fold{i}"
        weights = REPO / "runs" / "detect" / run / "weights" / "best.pt"
        if not a.eval_only:
            cmd = [sys.executable, str(REPO / "tools" / "train.py"), "--name", run, "--model", a.model, "--data", str(yml),
                   "--imgsz", str(a.imgsz), "--batch", str(a.batch), "--epochs", str(a.epochs),
                   "--patience", str(a.patience), "--device", a.device]
            if a.aug:
                cmd += ["--aug", a.aug]
            subprocess.run(cmd, check=True)
        if not weights.exists():
            raise SystemExit(f"Khong thay {weights}")
        m = evaluate(weights, yml, a.imgsz, a.batch, a.device)
        rows.append({"fold": i, "test_boards": "+".join(f["test"]), "val_boards": "+".join(f["val"]), **m})
        print(f"  fold{i} test: mAP50={m['mAP50']:.4f} mAP50-95={m['mAP50_95']:.4f} P={m['P']:.4f} R={m['R']:.4f}")

    if a.dry_run or not rows:
        return
    stats = summarize(rows)
    fields = ["fold", "test_boards", "val_boards", "P", "R", "mAP50", "mAP50_95"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows + stats:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r.get(k), float) else r.get(k, "")) for k in fields})
    note = "" if len(rows) == N_FOLDS else f"  CANH BAO: chi {len(rows)}/{N_FOLDS} fold, chua phai ket qua kiem dinh cheo day du."
    print(f"Da ghi {a.out} ({len(rows)} fold). Trung binh +- do lech chuan o hai dong cuoi.{note}")


if __name__ == "__main__":
    main()
