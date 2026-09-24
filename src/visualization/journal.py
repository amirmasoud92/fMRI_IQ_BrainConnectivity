"""
Figure style, palette, panel helpers and export for the journal figures.

Every figure is drawn on a fixed canvas and saved as PNG, SVG and PDF together with the data of each
panel and a dictionary recording the sources and hashes.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
FIG_ROOT = ROOT / "figures"

CM = 1 / 2.54
WIDTH = {"single": 8.7 * CM, "onehalf": 11.4 * CM, "double": 17.8 * CM}

# ---------------------------------------------------------------------------
# colour
# ---------------------------------------------------------------------------
PAL = {
    "ink": "#1B1F24",       # text, axes, reference lines
    "muted": "#56616C",     # secondary text
    "slate": "#6E7B88",     # neutral data (confounds, inconclusive)
    "mist": "#CDD4DB",      # backgrounds, reference bands
    "navy": "#1D3557",
    "cobalt": "#2F66B0",
    "cerulean": "#62AEE3",
    "teal": "#17A08C",
    "saffron": "#F0A83A",
    "coral": "#E0573A",
    "plum": "#8D4E91",
    "rose": "#C94F7C",
}

ARM = {"confounds": PAL["slate"], "fc": PAL["cobalt"], "stack": PAL["coral"], "morph": PAL["teal"],
       "fa": PAL["plum"], "amplitude": PAL["saffron"], "random": PAL["mist"]}
COLLECTION = {"ID1000": PAL["cobalt"], "PIOP1": PAL["teal"], "PIOP2": PAL["saffron"]}
VERDICT = {"established": PAL["coral"], "better": PAL["teal"], "equivalent": PAL["cerulean"],
           "inconclusive": PAL["slate"], "identical": PAL["mist"]}
SEX = {"male": PAL["cerulean"], "female": PAL["rose"]}

# Yeo et al. (2011) 7-network colours, the convention readers recognise on a cortical surface
NETWORK = {"Vis": "#781286", "SomMot": "#4682B4", "DorsAttn": "#00760E", "SalVentAttn": "#C43AFA",
           "Limbic": "#B5CF6B", "Cont": "#E69422", "Default": "#CD3E4E", "Subcortex": "#8C8C8C"}
NETWORK_LABEL = {"Vis": "Visual", "SomMot": "Somatomotor", "DorsAttn": "Dorsal attention",
                 "SalVentAttn": "Salience", "Limbic": "Limbic", "Cont": "Control", "Default": "Default",
                 "Subcortex": "Subcortex"}

_DIVERGING = ["#0E2A47", "#1F5A8C", "#4A8CC2", "#9CC4E4", "#DCEBF5", "#F7F5F0",
              "#F8DCC8", "#F0A882", "#DD6B45", "#AE3A2F", "#5C1A24"]
_SEQUENTIAL = ["#FFF8EC", "#FCE3B6", "#F7BE7A", "#EE8F4F", "#D65F3B", "#A33B3B", "#6B2645", "#2E1638"]
DIVERGING = LinearSegmentedColormap.from_list("ocean_ember", _DIVERGING, N=512)
SEQUENTIAL = LinearSegmentedColormap.from_list("ember", _SEQUENTIAL, N=512)
DIVERGING.set_bad("#EEF1F4")
SEQUENTIAL.set_bad("#EEF1F4")


def verdict_colour(verdict: str, delta: float) -> str:
    """Colour of a corrected comparison: established worse / better, equivalent, inconclusive."""
    if verdict == "established":
        return VERDICT["established"] if delta < 0 else VERDICT["better"]
    return VERDICT.get(verdict, PAL["slate"])


def apply_style() -> None:
    for cmap in (DIVERGING, SEQUENTIAL):
        if cmap.name not in mpl.colormaps:
            mpl.colormaps.register(cmap)
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial", "mathtext.default": "regular",
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "text.color": PAL["ink"], "axes.labelcolor": PAL["ink"], "axes.edgecolor": PAL["ink"],
        "xtick.color": PAL["ink"], "ytick.color": PAL["ink"],
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5, "lines.linewidth": 1.0, "patch.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlepad": 4, "axes.labelpad": 2,
        "legend.frameon": False, "legend.handlelength": 1.2, "legend.borderaxespad": 0.2,
        "savefig.dpi": 600, "figure.dpi": 150, "savefig.bbox": None, "savefig.pad_inches": 0,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "image.cmap": DIVERGING.name, "axes.unicode_minus": True, "svg.hashsalt": "aomic-figures",
    })


def panel_label(ax, letter: str, x: float = -0.14, y: float = 1.06) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


def fig_label(fig, letter: str, x: float, y: float) -> None:
    """Panel letter in figure coordinates (for schematic panels without axes)."""
    fig.text(x, y, letter, fontsize=9, fontweight="bold", va="top", ha="left")


def cell_label(fig, cell, letter: str, dx_pt: float = 0.0, dy_pt: float = 0.0):
    """Panel letter at the top-left corner of a gridspec cell (robust when the panel is not a single axes)."""
    from matplotlib.transforms import ScaledTranslation
    box = cell.get_position(fig)
    offset = ScaledTranslation(dx_pt / 72, dy_pt / 72, fig.dpi_scale_trans)
    return fig.text(box.x0, box.y1, letter, fontsize=9, fontweight="bold", va="bottom", ha="left",
                    transform=fig.transFigure + offset)


def p_text(p: float) -> str:
    if not np.isfinite(p):
        return "p = n/a"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def style_colorbar(cb, label: str, size: float = 6.0) -> None:
    cb.set_label(label, fontsize=size)
    cb.ax.tick_params(labelsize=size - 0.5, length=2, width=0.5)
    cb.outline.set_linewidth(0.4)


# ---------------------------------------------------------------------------
# canvas check and export
# ---------------------------------------------------------------------------
OVERFLOW_TOLERANCE_PT = 0.5


def canvas_overflow(fig) -> list[str]:
    """Describe every axes or figure-level text whose drawn extent leaves the canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    tol = OVERFLOW_TOLERANCE_PT * fig.dpi / 72
    problems = []

    def check(name, bb):
        over = {"left": -bb.x0, "right": bb.x1 - W, "bottom": -bb.y0, "top": bb.y1 - H}
        bad = {k: v * 72 / fig.dpi for k, v in over.items() if v > tol}
        if bad:
            problems.append(f"{name}: " + ", ".join(f"{k} +{v:.1f} pt" for k, v in bad.items()))

    for i, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue
        title = ax.get_title(loc="left") or ax.get_title() or ax.get_ylabel() or ax.get_xlabel()
        check(f"axes {i} ({title[:40]!r})", ax.get_tightbbox(renderer))
    for t in fig.texts:
        if t.get_visible() and t.get_text():
            check(f"figure text {t.get_text()[:40]!r}", t.get_window_extent(renderer))
    for leg in fig.legends:
        check("figure legend", leg.get_window_extent(renderer))
    return problems


def fit_to_canvas(fig, margin_pt: float = 1.5, max_iter: int = 8) -> None:
    """Shrink the layout just enough that every drawn artist lies inside the canvas.

    All top-level axes are mapped through one common affine transform per direction, so relative panel
    alignment is preserved (inset axes follow their parents). Text keeps its size, which is why this iterates.
    A layout that already fits is left untouched.
    """
    from matplotlib.transforms import Bbox
    W, H = fig.bbox.width, fig.bbox.height
    m = margin_pt * fig.dpi / 72
    tol = OVERFLOW_TOLERANCE_PT * fig.dpi / 72
    for _ in range(max_iter):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [ax.get_tightbbox(renderer) for ax in fig.axes if ax.get_visible()]
        boxes += [t.get_window_extent(renderer) for t in fig.texts if t.get_visible() and t.get_text()]
        boxes += [leg.get_window_extent(renderer) for leg in fig.legends]
        u = Bbox.union([b for b in boxes if b is not None and b.width > 0])
        if u.x0 >= m - tol and u.x1 <= W - m + tol and u.y0 >= m - tol and u.y1 <= H - m + tol:
            return
        nx0, nx1 = max(u.x0, m), min(u.x1, W - m)
        ny0, ny1 = max(u.y0, m), min(u.y1, H - m)
        sx, sy = (nx1 - nx0) / u.width, (ny1 - ny0) / u.height
        for ax in fig.axes:
            if ax.get_axes_locator() is not None:
                continue
            p = ax.get_position(original=True)
            x0 = (nx0 + (p.x0 * W - u.x0) * sx) / W
            x1 = (nx0 + (p.x1 * W - u.x0) * sx) / W
            y0 = (ny0 + (p.y0 * H - u.y0) * sy) / H
            y1 = (ny0 + (p.y1 * H - u.y0) * sy) / H
            ax.set_position([x0, y0, x1 - x0, y1 - y0])


FIGURE_CODE = ("scripts/figures", "src/visualization/journal.py", "src/visualization/brain.py")


def _git_commit() -> str:
    """Short HEAD, suffixed '-dirty' when the figure code differs from that commit."""
    out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True)
    commit = out.stdout.strip() or "unknown"
    status = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", *FIGURE_CODE],
                            capture_output=True, text=True)
    return f"{commit}-dirty" if status.stdout.strip() else commit


# byte-stable exports: no timestamps, fixed SVG element ids
_METADATA = {"png": None, "svg": {"Date": None}, "pdf": {"Creator": "scripts/figures", "CreationDate": None}}


@dataclass
class FigureRecord:
    """Collects panel data and writes the figure plus its data dictionary."""
    name: str                      # e.g. "Fig2"
    title: str
    claim: str
    panels: list = field(default_factory=list)
    sources: set = field(default_factory=set)

    @property
    def folder(self) -> Path:
        d = FIG_ROOT / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def source(self, rel_path: str) -> Path:
        """Register a locked input table and return its absolute path."""
        p = ROOT / rel_path
        if not p.exists():
            raise FileNotFoundError(rel_path)
        self.sources.add(rel_path)
        return p

    def data(self, panel: str, slug: str, df: pd.DataFrame, description: str,
             columns: dict[str, str] | None = None) -> pd.DataFrame:
        fn = f"{self.name}_{panel}_{slug}.csv"
        df.to_csv(self.folder / fn, index=False, lineterminator="\n")
        self.panels.append({"panel": panel, "file": fn, "description": description,
                            "columns": columns or {c: "" for c in df.columns}, "rows": len(df)})
        return df

    def save(self, fig, fit: bool = True) -> list[Path]:
        if fit:
            fit_to_canvas(fig)
        problems = canvas_overflow(fig)
        if problems:
            plt.close(fig)
            raise ValueError(f"{self.name}: artists outside the {fig.get_figwidth() / CM:.1f} cm canvas\n  "
                             + "\n  ".join(problems))
        paths = []
        for ext in ("png", "svg", "pdf"):
            p = self.folder / f"{self.name}.{ext}"
            fig.savefig(p, dpi=600 if ext == "png" else None, metadata=_METADATA[ext])
            paths.append(p)
        self._write_dictionary(paths, fig.get_size_inches() / CM)
        produced = {p["file"] for p in self.panels}
        for stale in self.folder.glob(f"{self.name}_*.csv"):     # panel files from an earlier layout
            if stale.name not in produced:
                stale.unlink()
        plt.close(fig)
        return paths

    def _write_dictionary(self, images: list[Path], size_cm) -> None:
        from src.io_utils import sha256_file
        lines = [f"# {self.name}: {self.title}", "",
                 f"**Claim.** {self.claim}", "",
                 f"Generated {date.today().isoformat()} at commit `{_git_commit()}` by "
                 f"`scripts/figures/{self.name.lower()}.py` (a `-dirty` commit means the figure code was not yet "
                 f"committed). Canvas {size_cm[0]:.1f} x {size_cm[1]:.1f} cm. Image files are not versioned; their "
                 f"hashes are:", "",
                 "| file | sha256 |", "|---|---|"]
        lines += [f"| `{p.name}` | `{sha256_file(p)[:16]}…` |" for p in images]
        lines += ["", "PNG at 600 dpi; SVG keeps text as text; PDF embeds TrueType fonts.", "", "## Panel data", ""]
        for p in self.panels:
            lines += [f"### Panel {p['panel']} — `{p['file']}` ({p['rows']} rows)", "", p["description"], "",
                      "| column | meaning |", "|---|---|"]
            lines += [f"| `{c}` | {m} |" for c, m in p["columns"].items()]
            lines.append("")
        lines += ["## Locked source tables", "", "| file | sha256 |", "|---|---|"]
        lines += [f"| `{s}` | `{sha256_file(ROOT / s)[:16]}…` |" for s in sorted(self.sources)]
        (self.folder / f"{self.name}_data.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# small plotting helpers
# ---------------------------------------------------------------------------
def forest(ax, labels, est, lo, hi, colors, ref=0.0, markersize=3.2, height=0.7):
    """Horizontal point-and-interval plot, first label at the top."""
    y = np.arange(len(labels))[::-1]
    for yi, e, l, h, c in zip(y, est, lo, hi, colors):
        ax.plot([l, h], [yi, yi], color=c, lw=1.0, solid_capstyle="butt")
        ax.plot(e, yi, "o", color=c, ms=markersize, mec="white", mew=0.4, zorder=3)
    ax.axvline(ref, color=PAL["ink"], lw=0.6, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return y


def strip(ax, x, values, color, width=0.28, size=6, alpha=0.75, seed=0, zorder=2):
    """Jittered fold-level dots."""
    rng = np.random.default_rng(seed)
    xs = x + rng.uniform(-width / 2, width / 2, len(values))
    ax.scatter(xs, values, s=size, color=color, alpha=alpha, lw=0, zorder=zorder)
