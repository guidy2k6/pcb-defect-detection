"""UI smoke tests; skipped when gradio is not installed."""
import pytest

pytest.importorskip("gradio")

from demo.app import build_ui, discover_examples, verdict_md  # noqa: E402
from demo.service import PCBInspector  # noqa: E402
from tests.test_service import FakeClassifier, FakeDetector, settings  # noqa: E402


def test_build_ui_and_verdict(tmp_path):
    insp = PCBInspector(settings(tmp_path), detector=FakeDetector(), classifier=FakeClassifier())
    assert build_ui(insp) is not None
    assert build_ui(insp, startup_error="thiếu trọng số") is not None


def test_discover_examples_pairs_by_board(tmp_path):
    for name in ("08_short_01.jpg", "08_spur_02.jpg", "12_short_01.jpg", "template_08.jpg"):
        (tmp_path / name).write_bytes(b"x")
    defects, pairs = discover_examples(tmp_path)
    assert len(defects) == 3
    assert len(pairs) == 2 and all(p[1].endswith("template_08.jpg") for p in pairs)
    assert discover_examples(tmp_path / "missing") == ([], [])
