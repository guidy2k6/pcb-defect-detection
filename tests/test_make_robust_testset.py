import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def test_make_robust_train_split_keeps_clean_images_and_transforms_labels(tmp_path):
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    images = {}
    source_label = "2 0.3 0.35 0.2 0.2\n"
    for index in range(12):
        image_path = image_dir / f"{index:02}.png"
        image = np.random.default_rng(index).integers(0, 256, (100, 120, 3), dtype=np.uint8)
        assert cv2.imwrite(str(image_path), image)
        (label_dir / f"{index:02}.txt").write_text(source_label, encoding="utf-8")
        images[image_path.name] = image
    config = tmp_path / "data.yaml"
    config.write_text(
        "path: .\ntrain: images/train\nval: images/train\ntest: images/train\n"
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
            "--split",
            "train",
            "--clean-ratio",
            "0.5",
            "--seed",
            "7",
        ],
        check=True,
        cwd=REPO,
    )

    output_images = output / "images" / "train"
    output_labels = output / "labels" / "train"
    assert len(list(output_images.iterdir())) == len(images)
    assert len(list(output_labels.iterdir())) == len(images)
    clean_count = 0
    changed_count = 0
    for name, source in images.items():
        result = cv2.imread(str(output_images / name))
        label = list(map(float, (output_labels / f"{Path(name).stem}.txt").read_text().split()))
        if result.shape == source.shape and np.array_equal(result, source):
            clean_count += 1
            assert np.allclose(label, [2, 0.3, 0.35, 0.2, 0.2], atol=1e-6)
        else:
            changed_count += 1
            assert result.shape[0] > source.shape[0]
            assert result.shape[1] > source.shape[1]
            assert label[0] == 2
            assert all(0 <= coordinate <= 1 for coordinate in label[1:])
            assert not np.allclose(label[1:], [0.3, 0.35, 0.2, 0.2], atol=1e-3)

    assert clean_count > 0
    assert changed_count > 0


def test_make_robust_default_generates_train_val_and_test(tmp_path):
    for split in ("train", "val", "test"):
        image_dir = tmp_path / "images" / split
        label_dir = tmp_path / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        assert cv2.imwrite(
            str(image_dir / "board.png"),
            np.full((16, 20, 3), 127, dtype=np.uint8),
        )
        (label_dir / "board.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    config = tmp_path / "data.yaml"
    config.write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
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
            "--clean-ratio",
            "1",
        ],
        check=True,
        cwd=REPO,
    )

    for split in ("train", "val", "test"):
        assert (output / "images" / split / "board.png").is_file()
        assert (output / "labels" / split / "board.txt").is_file()
