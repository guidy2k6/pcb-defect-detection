import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def test_make_canary_writes_crops_per_class(tmp_path):
    img_dir, lab_dir = tmp_path / "images" / "val", tmp_path / "labels" / "val"
    img_dir.mkdir(parents=True), lab_dir.mkdir(parents=True)
    names = ["Missing Hole", "Mouse Bite", "Open Circuit", "Short", "Spur", "Spurious Copper"]
    for i in range(3):
        cv2.imwrite(str(img_dir / f"07_x_{i}.jpg"), np.random.default_rng(i).integers(0, 255, (400, 600, 3), dtype=np.uint8))
        (lab_dir / f"07_x_{i}.txt").write_text("".join(f"{c} {0.2 + 0.1 * c:.2f} 0.5 0.1 0.1\n" for c in range(6)))
    (tmp_path / "data.yaml").write_text("path: .\ntrain: images/val\nval: images/val\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    out = tmp_path / "canary"
    subprocess.run([sys.executable, str(REPO / "tools" / "make_canary.py"), "--data", str(tmp_path / "data.yaml"), "--per-class", "2",
                    "--out", str(out)], check=True, cwd=REPO / "tools")
    files = sorted(p.name for p in out.glob("*.png"))
    assert files == sorted(f"{n.lower().replace(' ', '_')}_{k}.png" for n in names for k in (1, 2))
    assert cv2.imread(str(out / files[0])).shape == (128, 128, 3)
