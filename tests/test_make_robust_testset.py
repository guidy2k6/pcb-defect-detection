import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def test_make_robust_testset_transforms_boxes_and_preserves_blur_labels(tmp_path):
    image_dir = tmp_path / "images" / "test"
    label_dir = tmp_path / "labels" / "test"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image_path = image_dir / "07_board.png"
    image = np.full((100, 120, 3), 127, dtype=np.uint8)
    image[:10, :10] = (255, 0, 0)
    cv2.imwrite(str(image_path), image)
    source_label = "2 0.5 0.5 0.2 0.2\n"
    (label_dir / "07_board.txt").write_text(source_label, encoding="utf-8")
    config = tmp_path / "data.yaml"
    config.write_text(
        "path: .\ntrain: images/test\nval: images/test\ntest: images/test\n"
        "names:\n  0: defect\n",
        encoding="utf-8",
    )
    output = tmp_path / "robust"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "data" / "io" / "make_robust_testset.py"),
            "--data",
            str(config),
            "--out",
            str(output),
            "--rotate-angles",
            "90",
            "10",
            "--blur-kernels",
            "3",
        ],
        check=True,
        cwd=REPO,
    )

    rotated = output / "labels" / "rotate_p90" / "07_board__rotate_p90.txt"
    tilted = output / "labels" / "tilt_top" / "07_board__tilt_top.txt"
    blur = output / "labels" / "blur_k3" / "07_board__blur_k3.txt"
    rotated_box = list(map(float, rotated.read_text(encoding="utf-8").split()))
    tilted_box = list(map(float, tilted.read_text(encoding="utf-8").split()))
    assert rotated_box[0] == 2
    assert np.allclose(rotated_box[1:], [0.5, 0.5, 0.2, 0.2], atol=1e-5)
    assert tilted_box[0] == 2
    assert all(0 <= coordinate <= 1 for coordinate in tilted_box[1:])
    assert not np.allclose(tilted_box[1:], [0.5, 0.5, 0.2, 0.2])
    assert np.allclose(
        list(map(float, blur.read_text(encoding="utf-8").split())),
        [2, 0.5, 0.5, 0.2, 0.2],
        atol=1e-6,
    )
    rotated_image = cv2.imread(str(output / "images" / "rotate_p90" / "07_board__rotate_p90.png"))
    assert rotated_image.shape[:2] == (120, 100)
    assert np.count_nonzero(
        (rotated_image[..., 0] > 250)
        & (rotated_image[..., 1] < 5)
        & (rotated_image[..., 2] < 5)
    ) >= 90
    angled_image = cv2.imread(str(output / "images" / "rotate_p10" / "07_board__rotate_p10.png"))
    assert angled_image.shape[0] > 100 and angled_image.shape[1] > 120
    assert np.count_nonzero(
        (angled_image[..., 0] > 250)
        & (angled_image[..., 1] < 5)
        & (angled_image[..., 2] < 5)
    ) >= 70
    tilted_image = cv2.imread(str(output / "images" / "tilt_top" / "07_board__tilt_top.png"))
    assert tilted_image is not None
    assert np.all(tilted_image[0, 0] == 0)
    tilted_bottom_image = cv2.imread(
        str(output / "images" / "tilt_bottom" / "07_board__tilt_bottom.png")
    )
    assert tilted_bottom_image is not None
    assert np.all(tilted_bottom_image[-1, 0] == 0)
