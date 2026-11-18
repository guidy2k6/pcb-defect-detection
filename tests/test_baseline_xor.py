"""Tests for baseline XOR feature matching and alignment."""

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import baseline_xor  # noqa: E402


def _write_image(path: Path) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_process_pair_finds_homography_on_small_images_and_scales_it_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_path = tmp_path / "baseline.png"
    defect_path = tmp_path / "defect.png"
    _write_image(baseline_path)
    _write_image(defect_path)
    baseline_xor._BASELINE_FEATURE_CACHE.clear()

    keypoints = [
        SimpleNamespace(pt=(float(index), float(index + 1)))
        for index in range(10)
    ]
    descriptors = np.ones((10, 128), dtype=np.float32)

    class FakeSift:
        def detectAndCompute(self, image: np.ndarray, mask: object) -> tuple:
            assert image.shape == (32, 32)
            return keypoints, descriptors

    class FakeMatcher:
        def knnMatch(self, query: np.ndarray, train: np.ndarray, k: int) -> list:
            assert k == 2
            return [
                [
                    SimpleNamespace(queryIdx=i, trainIdx=i, distance=1.0),
                    SimpleNamespace(queryIdx=i, trainIdx=i, distance=2.0),
                ]
                for i in range(10)
            ]

    observed_points: dict[str, np.ndarray] = {}

    def fake_find_homography(
        source: np.ndarray,
        target: np.ndarray,
        method: int,
        threshold: float,
    ) -> tuple[np.ndarray, None]:
        observed_points["source"] = source
        observed_points["target"] = target
        return np.array(
            [[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]]
        ), None

    monkeypatch.setattr(cv2, "SIFT_create", lambda: FakeSift())
    monkeypatch.setattr(cv2, "FlannBasedMatcher", lambda *args: FakeMatcher())
    monkeypatch.setattr(cv2, "findHomography", fake_find_homography)

    _, _, homography = baseline_xor.process_pair(baseline_path, defect_path)

    assert observed_points["source"].shape == (10, 1, 2)
    assert np.allclose(observed_points["source"][0, 0], [0, 1])
    assert np.allclose(
        homography,
        [[1.0, 0.0, 4.0], [0.0, 1.0, 6.0], [0.0, 0.0, 1.0]],
    )


def test_process_pair_skips_matching_when_descriptor_count_is_too_small(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_path = tmp_path / "baseline.png"
    defect_path = tmp_path / "defect.png"
    _write_image(baseline_path)
    _write_image(defect_path)

    class FakeSift:
        def detectAndCompute(self, image: np.ndarray, mask: object) -> tuple:
            return [], np.ones((10, 128), dtype=np.float32)

    def unexpected_matcher(*args: object) -> None:
        pytest.fail("matcher should not run when fewer than ten descriptors exist")

    monkeypatch.setattr(cv2, "SIFT_create", lambda: FakeSift())
    monkeypatch.setattr(
        baseline_xor,
        "_baseline_sift_features",
        lambda *args: ((), np.ones((9, 128), dtype=np.float32)),
    )
    monkeypatch.setattr(cv2, "FlannBasedMatcher", unexpected_matcher)

    with pytest.raises(ValueError, match="Không đủ điểm khớp"):
        baseline_xor.process_pair(baseline_path, defect_path)
