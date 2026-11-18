"""Ve duong cong chiu nhieu va xuat bang LaTeX tu results/robustness.csv.

Moi con so trong hinh va bang lay thang tu file csv, khong go tay.

    python tools/plot_robustness.py --split val          # chon mo hinh
    python tools/plot_robustness.py --split test         # so cuoi (sau khi chot mo hinh)

Dau ra: results/robustness_<split>.png, reports/paper/bang_chiunhieu_<split>.tex
Trong bao cao:  \\input{bang_chiunhieu_test.tex}   va   \\includegraphics{robustness_test.png}
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

LABEL = {"clean": "Gốc", "rot15": "Xoay 15°", "rot45": "Xoay 45°", "rot90": "Xoay 90°", "rot180": "Xoay 180°",
         "tilt5": "Nghiêng 5%", "tilt10": "Nghiêng 10%", "tilt15": "Nghiêng 15%",
         "blur_s1": "Blur σ=1", "blur_s2": "Blur σ=2", "blur_s3": "Blur σ=3", "blur_s5": "Blur σ=5",
         "motion15": "Motion blur 15 px", "dark50": "Tối 50%"}
PANELS = [("rot", "Xoay (độ)"), ("tilt", "Nghiêng (độ co cạnh trên, %)"), ("blur", "Gaussian blur (σ, pixel)")]


def load(path: Path, split: str):
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    if not rows:
        raise SystemExit(f"{path} không có dòng nào của split '{split}'.")
    return rows


def series(rows, model, family):
    """[(level, mAP50, R)] cho một nhóm, thêm điểm 'gốc' ở level 0, sắp theo level."""
    clean = next((r for r in rows if r["model"] == model and r["perturbation"] == "clean"), None)
    pts = [(float(r["level"]) * (100 if family == "tilt" else 1), float(r["mAP50"]), float(r["R"]))
           for r in rows if r["model"] == model and r["family"] == family]
    if clean:
        pts.append((0.0, float(clean["mAP50"]), float(clean["R"])))
    return sorted(pts)


def make_figure(rows, out_png: Path, split: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({r["model"] for r in rows})
    colors = {m: c for m, c in zip(models, ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"])}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (fam, xlabel) in zip(axes.flat[:3], PANELS):
        for m in models:
            pts = series(rows, m, fam)
            if len(pts) < 2:
                continue
            xs = [p[0] for p in pts]
            ax.plot(xs, [p[1] for p in pts], "o-", color=colors[m], label=f"{m} · mAP50")
            ax.plot(xs, [p[2] for p in pts], "s--", color=colors[m], alpha=0.7, label=f"{m} · recall")
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    ax = axes.flat[3]
    names = [n for n in ("clean", "motion15", "dark50") if any(r["perturbation"] == n for r in rows)]
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        vals = []
        for n in names:
            r = next((r for r in rows if r["model"] == m and r["perturbation"] == n), None)
            vals.append(float(r["mAP50"]) if r else 0.0)
        ax.bar([k + i * width for k in range(len(names))], vals, width, color=colors[m], label=f"{m} · mAP50")
    ax.set_xticks([k + width * (len(models) - 1) / 2 for k in range(len(names))])
    ax.set_xticklabels([LABEL[n] for n in names])
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3, axis="y")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), frameon=False)
    fig.suptitle(f"Độ bền với xoay, nghiêng, mờ, tối (tập {split})")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


def make_table(rows, out_tex: Path, split: str):
    models = sorted({r["model"] for r in rows})
    perts = [n for n in LABEL if any(r["perturbation"] == n for r in rows)]
    by = {(r["model"], r["perturbation"]): r for r in rows}
    cols = "l|c|" + "|".join("cc" for _ in models)
    head1 = " & ".join([r"\textbf{Điều kiện}", r"\textbf{Độ khít nhãn}"] + [rf"\multicolumn{{2}}{{c}}{{\texttt{{{tex_escape(m)}}}}}" for m in models])
    head2 = " & ".join(["", ""] + ["mAP50 & $R$"] * len(models))
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             rf"\caption{{Độ bền theo điều kiện nhiễu, tập {split} (\texttt{{results/robustness.csv}}). "
             r"Độ khít nhãn là tỉ lệ diện tích box gốc so với box sau biến đổi; dưới 1 nghĩa là nhãn bị nở ra nên mAP50 bị chặn trên.}"
             r"\label{tab:chiunhieu_" + split + "}",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule", head1 + r" \\", head2 + r" \\ \midrule"]
    for n in perts:
        first = next((by[(m, n)] for m in models if (m, n) in by), None)
        cells = [tex_escape(LABEL[n]), f"{float(first['label_tightness']):.2f}" if first else "--"]
        for m in models:
            r = by.get((m, n))
            cells += [f"{float(r['mAP50']):.4f}", f"{float(r['R']):.4f}"] if r else ["--", "--"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(REPO / "results" / "robustness.csv"))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--png", default=None)
    ap.add_argument("--tex", default=None)
    a = ap.parse_args()
    rows = load(Path(a.csv), a.split)
    png = Path(a.png or REPO / "results" / f"robustness_{a.split}.png")
    tex = Path(a.tex or REPO / "reports" / "paper" / f"bang_chiunhieu_{a.split}.tex")
    make_figure(rows, png, a.split)
    make_table(rows, tex, a.split)
    print(f"Đã ghi {png} và {tex}")


if __name__ == "__main__":
    main()
