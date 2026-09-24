"""Tests for the figure style, export and data dictionaries."""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization import journal as J


def _record(tmp: Path) -> J.FigureRecord:
    J.FIG_ROOT = tmp
    rec = J.FigureRecord(name="FigT", title="test", claim="a claim")
    return rec


def _draw(rec: J.FigureRecord):
    J.apply_style()
    fig, ax = plt.subplots(figsize=(2, 2))
    x = np.linspace(0, 1, 20)
    ax.plot(x, x ** 2)
    ax.set_title("t")
    rec.data("A", "curve", pd.DataFrame({"x": x, "y": x ** 2}), "a curve", {"x": "input", "y": "square"})
    return rec.save(fig)


def test_verdict_colour():
    assert J.verdict_colour("established", -0.1) == J.VERDICT["established"]
    assert J.verdict_colour("established", 0.1) == J.VERDICT["better"]
    assert J.verdict_colour("equivalent", 0.0) == J.VERDICT["equivalent"]
    assert J.verdict_colour("inconclusive", 0.3) == J.VERDICT["inconclusive"]


def _lightness(rgb: np.ndarray) -> np.ndarray:
    """CIELAB L* of sRGB colours in [0, 1] (D65)."""
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    y = c @ np.array([0.2126729, 0.7151522, 0.0721750])
    return np.where(y > (6 / 29) ** 3, 116 * np.cbrt(y) - 16, y * (29 / 3) ** 3)


def test_colormap_lightness_is_monotonic():
    x = np.linspace(0, 1, 257)
    seq = _lightness(J.SEQUENTIAL(x)[:, :3])
    assert np.all(np.diff(seq) < 0), "sequential map must darken monotonically"
    div = _lightness(J.DIVERGING(x)[:, :3])
    mid = len(x) // 2
    assert np.all(np.diff(div[:mid + 1]) > 0) and np.all(np.diff(div[mid:]) < 0), "diverging map must peak at 0"
    assert abs(div[0] - div[-1]) < 6, "diverging arms should end at similar lightness"


def test_overflowing_label_is_detected_and_fitted():
    J.apply_style()
    fig, ax = plt.subplots(figsize=(J.WIDTH["single"], 2.0))
    fig.subplots_adjust(left=0.01)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["a very long tick label that leaves the canvas", "short"])
    assert J.canvas_overflow(fig), "a label outside the canvas must be reported"
    J.fit_to_canvas(fig)
    assert not J.canvas_overflow(fig), "fitting must bring every artist inside the canvas"
    plt.close(fig)


def test_p_text():
    assert J.p_text(0.0004) == "p < 0.001"
    assert J.p_text(0.0071) == "p = 0.007"
    assert J.p_text(0.2345) == "p = 0.23"
    assert J.p_text(float("nan")) == "p = n/a"


def test_save_writes_all_formats_data_and_dictionary():
    old = J.FIG_ROOT
    try:
        with tempfile.TemporaryDirectory() as d:
            rec = _record(Path(d))
            paths = _draw(rec)
            assert [p.suffix for p in paths] == [".png", ".svg", ".pdf"]
            assert all(p.stat().st_size > 0 for p in paths)
            csv = Path(d) / "FigT" / "FigT_A_curve.csv"
            assert pd.read_csv(csv).shape == (20, 2)
            md = (Path(d) / "FigT" / "FigT_data.md").read_text(encoding="utf-8")
            assert "FigT_A_curve.csv" in md and "| `x` | input |" in md
            for p in paths:
                digest = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                assert f"| `{p.name}` | `{digest}…` |" in md, p.name
    finally:
        J.FIG_ROOT = old


def test_vector_exports_are_byte_stable():
    old = J.FIG_ROOT
    try:
        digests = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as d:
                paths = _draw(_record(Path(d)))
                digests.append([hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])
        assert digests[0] == digests[1], "PNG/SVG/PDF differ between identical runs"
    finally:
        J.FIG_ROOT = old


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:                                   # noqa: BLE001
            import traceback; traceback.print_exc()
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
