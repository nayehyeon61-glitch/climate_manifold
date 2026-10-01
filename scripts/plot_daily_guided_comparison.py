"""Paper figures for the daily guided Transformer comparison.

Usage: python scripts/plot_daily_guided_comparison.py <run dir or metrics dir> <output dir>
Requires matplotlib and pandas (not project dependencies)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

RUN = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

for f in Path("/usr/share/fonts/urw-base35").glob("NimbusSans-*.otf"):
    fm.fontManager.addfont(str(f))
plt.rcParams.update({
    "font.family": ["Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
GRID = "#e1e0d9"
ARMS = [  # (id, label, color)
    ("guided", "Guided (learned)", "#2a78d6"),
    ("raw", "Raw", "#eb6834"),
    ("latent", "Latent", "#1baf7a"),
    ("guided_zero", "Guided (zero)", "#eda100"),
]
LABEL = {a: l for a, l, _ in ARMS}
COLOR = {a: c for a, _, c in ARMS}
VARS = [("msl", "MSL", "Pa"), ("t2m", "T2M", "K"), ("u10", "U10", "m s$^{-1}$"), ("v10", "V10", "m s$^{-1}$")]
LEADS = [24, 48, 72, 96, 120]
SEEDS = [7, 19, 43]
COL_W, FULL_W = 3.35, 6.9  # single / double column width, inches


def arm_of(row):
    if row.bridge != "guided":
        return row.bridge
    return "guided" if row.guide_mode == "learned" else "guided_zero"


def load(split):
    suf = ".test" if split == "test" else ""
    c = pd.read_csv(RUN / f"comparison{suf}.csv")
    c["arm"] = c.apply(arm_of, axis=1)
    sha = {s: a for s, a in zip(c.representation_sha256, c.arm) if isinstance(s, str)}
    d = pd.read_csv(RUN / f"comparison{suf}.climode.csv")
    d["arm"] = [sha.get(s, b) if b == "guided" else b for s, b in zip(d.representation_sha256, d.bridge)]
    return c, d


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)


def fig_overall(c, split):
    fig, ax = plt.subplots(figsize=(COL_W, 1.55))
    for i, (a, lab, col) in enumerate(ARMS):
        v = c[c.arm == a].normalized_rmse.to_numpy()
        y = len(ARMS) - 1 - i
        ax.hlines(y, v.min(), v.max(), color=col, lw=1.2, alpha=0.45, zorder=1)
        ax.scatter(v, [y] * len(v), s=16, color=col, edgecolor="white", linewidth=0.6, zorder=3)
        ax.vlines(v.mean(), y - 0.28, y + 0.28, color=col, lw=1.8, zorder=2)
        ax.text(1.01, y, f"{v.mean():.4f} ± {v.std(ddof=1):.4f}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=6.8, color="#0b0b0b", fontweight="bold" if a == "guided" else "normal")
    ax.set_yticks(range(len(ARMS)), [l for _, l, _ in ARMS][::-1])
    ax.set_ylim(-0.6, len(ARMS) - 0.4)
    ax.set_xlabel(f"Normalized RMSE ({split}; lower is better)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, f"fig1_overall_nrmse_{split}")


def skill_table(d, base):
    p = d.pivot_table(index=["variable", "lead_hours", "seed"], columns="arm", values="rmse")
    sk = (1 - p["guided"] / p[base]) * 100
    g = sk.groupby(["variable", "lead_hours"])
    mean = g.mean().unstack()
    agree = g.apply(lambda s: (s > 0).all() or (s < 0).all()).unstack()
    order = [v for v, _, _ in VARS]
    return mean.loc[order, LEADS], agree.loc[order, LEADS]


def fig_skill(d, split):
    cmap = LinearSegmentedColormap.from_list("div", ["#c03232", "#e88a89", "#f0efec", "#86b6ef", "#1c5cab"])
    bases = [("raw", "vs. Raw"), ("guided_zero", "vs. Guided (zero)"), ("latent", "vs. Latent")]
    tabs = [skill_table(d, b) for b, _ in bases]
    vmax = max(np.ceil(max(np.abs(t[0].to_numpy()).max() for t in tabs)), 1)
    norm = TwoSlopeNorm(0, -vmax, vmax)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 1.75), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, (b, title), (m, agree) in zip(axes, bases, tabs):
        im = ax.imshow(m.to_numpy(), cmap=cmap, norm=norm, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.iat[i, j]
                ax.text(j, i, f"{v:+.1f}".replace("-", "\u2212"), ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) / vmax > 0.6 else "#0b0b0b",
                        fontweight="bold" if agree.iat[i, j] else "normal")
        ax.set_xticks(range(len(LEADS)), [f"{l}" for l in LEADS])
        ax.set_yticks(range(len(VARS)), [lab for _, lab, _ in VARS])
        ax.set_xlabel("Lead time (h)")
        ax.set_title(title, pad=3)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-.5, len(LEADS)), minor=True)
        ax.set_yticks(np.arange(-.5, len(VARS)), minor=True)
        ax.grid(which="minor", color="white", lw=1.2)
        ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015)
    cb.set_label("RMSE reduction by\nGuided (learned) (%)", fontsize=7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=2)
    save(fig, f"fig2_rmse_skill_heatmap_{split}")


def fig_lead(d, split):
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 3.6), sharex=True, gridspec_kw={"hspace": 0.28, "wspace": 0.28})
    for ax, (v, lab, unit) in zip(axes.flat, VARS):
        for a, alab, col in ARMS[::-1]:
            s = d[(d.arm == a) & (d.variable == v)].groupby("lead_hours").rmse
            m, lo, hi = s.mean(), s.min(), s.max()
            ax.fill_between(m.index, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(m.index, m, color=col, lw=1.6 if a == "guided" else 1.1, marker="o", ms=2.8,
                    label=alab, zorder=4 if a == "guided" else 3)
        ax.set_title(lab, loc="left", fontweight="bold")
        ax.set_ylabel(f"RMSE ({unit})")
        ax.set_xticks(LEADS)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel("Lead time (h)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.93))
    fig.subplots_adjust(top=0.86)
    save(fig, f"fig3_rmse_by_lead_{split}")


def fig_curves():
    rows = []
    for f in RUN.glob("transformer-*-seed*.metrics.json"):
        arm, seed = f.name[len("transformer-"):-len(".metrics.json")].rsplit("-seed", 1)
        for e in json.loads(f.read_text()):
            rows.append((arm, int(seed), e["epoch"], e["selection"]["state_mse"]))
    df = pd.DataFrame(rows, columns=["arm", "seed", "epoch", "mse"])
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    for a, lab, col in ARMS[::-1]:
        g = df[df.arm == a].groupby("epoch").mse
        ax.fill_between(g.mean().index, g.min(), g.max(), color=col, alpha=0.15, lw=0)
        ax.plot(g.mean().index, g.mean(), color=col, lw=1.6 if a == "guided" else 1.1, label=lab,
                zorder=4 if a == "guided" else 3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Selection state MSE")
    ax.set_xlim(1, df.epoch.max())
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], frameon=False, loc="upper right")
    save(fig, "fig4_training_curves")


for split in ("test", "validation"):
    c, d = load(split)
    fig_overall(c, split)
    fig_skill(d, split)
    fig_lead(d, split)
fig_curves()
print("\n".join(sorted(p.name for p in OUT.iterdir())))
