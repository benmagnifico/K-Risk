#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Risk combined overview figure (replaces former Fig. 5/6/7).

Each panel is a pure draw_*() routine carrying NO title, NO grey sub-caption and
NO panel letter, so it can be (i) assembled into one combined figure (panel
letters added only there, for the manuscript) and (ii) exported on its own as
png + pdf for slide-deck re-composition.

Panels
  a  Source x risk-level, grouped by environment  [Fig5-left + Fig5-right + Fig6 env]
  b  Per-source behaviour-frame breakdown (HighD / ExpresswayA / FreewayB), log x
  c  Agent composition: car vs non-car (top) + non-car class counts (bottom)
  d  Speed distribution per scenario (ridgeline, task5 histograms)
  e  Near-collision signature: speed x TTC joint density + marginals (HighD, full)
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

# ----------------------------------------------------------------------------
# Publication rcParams (editable SVG/PDF text, Nature sans)
# ----------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
# AD_Datasets repo root (HERE = .../AD_Datasets/K-Risk_data&pipeline/K-Risk_pipeline)
_ROOT = os.path.dirname(os.path.dirname(HERE))

# CSV inputs (task4/task5) live under Datasets-Jingguang; fall back to HERE.
DATADIR = next(
    (d for d in (HERE, os.path.join(_ROOT, "Datasets-Jingguang", "Figure",
                                     "fig_statistic"))
     if os.path.exists(os.path.join(
         d, "task5_speed_distribution_by_dataset_hist.csv"))),
    HERE)

# Combined manuscript figure -> existing LaTeX project dir.
OUTDIR = os.path.join(_ROOT, "LaTeX", "K_Risk_0620")
PANELDIR = os.path.join(OUTDIR, "panels")
# Individual panels (enlarged text, PDF only) -> ImageGen/fig_statistic.
INDIVDIR = os.path.join(_ROOT, "ImageGen", "fig_statistic")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(PANELDIR, exist_ok=True)
os.makedirs(INDIVDIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Font-size scaling.  All in-panel hard-coded font sizes go through fs() so the
# individual slide-deck panels can be enlarged (text close to the combined-figure
# panel-letter size, ~10 pt) without disturbing the tightly-laid-out combined
# figure, which keeps _FONT_SCALE == 1.0.
# ----------------------------------------------------------------------------
_FONT_SCALE = 1.0

# Base values of the font-related rcParams (combined-figure sizes).
_BASE_RC = {
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
}


def fs(x):
    return x * _FONT_SCALE


def _apply_font_scale(scale):
    """Set the global text scale and rescale the font rcParams accordingly."""
    global _FONT_SCALE
    _FONT_SCALE = scale
    mpl.rcParams.update({k: v * scale for k, v in _BASE_RC.items()})

# ----------------------------------------------------------------------------
# Palette  (low-saturation, consistent across panels)
# ----------------------------------------------------------------------------
RISK = {"Moderate": "#C6D3E3", "High": "#E7B98F", "Extreme": "#B5524E"}
ENV = {
    "Highway":      "#4C72A8",   # high-speed highway (HighD, NGSIM)
    "UrbanFreeway": "#4E9D9B",   # urban freeway (ExpresswayA, FreewayB)
    "Intersection": "#9A6FB0",   # urban intersection (InD)
    "Roundabout":   "#76A86B",   # roundabout (RounD)
}
AV_GREY   = "#A9ACB5"
CAR_GREY  = "#D2D2D2"
HEAVY     = "#8A8D99"            # heavy / special non-car vehicles
VRU_WARM  = "#B5524E"            # strict VRU accent
KRISK_C   = "#B5524E"            # K-Risk (warm signal)
ORIG_C    = "#5B7FB4"            # Original (neutral reference)
INK       = "#2B2B2B"

KRISK_CMAP = LinearSegmentedColormap.from_list(
    "krisk_warm", ["#FFFFFF", "#F4DAD2", "#E2A89A", "#C8736A", "#9E3A37"])


def _load(name):
    return pd.read_csv(os.path.join(DATADIR, name))


# ============================================================================
# Panel a  --  Source x risk-level, grouped by environment
# ============================================================================
def draw_a(ax):
    rows = [
        ("HighD",       "Highway",      7026, 1781,   0),
        ("NGSIM",       "Highway",      2205,  556,   0),
        ("FreewayB",    "UrbanFreeway", 3572, 1008, 724),
        ("ExpresswayA", "UrbanFreeway", 2631, 1576, 312),
        ("InD",         "Intersection", 4215,  469,   0),
        ("RounD",       "Roundabout",   2477,  276,   0),
        ("AV*",         "AV",              0,    0,   0),
    ]
    AV_TOTAL = 2570

    y_pos, labels, env_groups = [], [], []
    cur, prev_env = 0.0, None
    for src, env, *_ in rows:
        if prev_env is not None and env != prev_env:
            cur += 0.9
        y_pos.append(cur); labels.append(src); env_groups.append(env)
        cur += 1.0; prev_env = env
    y_pos = np.array(y_pos)
    ymax = y_pos.max()
    y_plot = ymax - y_pos
    bar_h = 0.72

    for yp, (src, env, mod, high, ext) in zip(y_plot, rows):
        if env == "AV":
            ax.barh(yp, AV_TOTAL, height=bar_h, color=AV_GREY,
                    edgecolor="white", linewidth=0.6)
            ax.text(AV_TOTAL + 120, yp, f"{AV_TOTAL:,}", va="center",
                    ha="left", fontsize=fs(5.6), color=INK)
            continue
        left = 0
        for key, val in [("Moderate", mod), ("High", high), ("Extreme", ext)]:
            if val <= 0:
                continue
            ax.barh(yp, val, left=left, height=bar_h, color=RISK[key],
                    edgecolor="white", linewidth=0.6)
            left += val
        ax.text(left + 120, yp, f"{left:,}", va="center", ha="left",
                fontsize=fs(5.6), color=INK)

    ax.set_yticks(y_plot)
    ytexts = ax.set_yticklabels(labels, fontsize=fs(6.4))
    for t, env in zip(ytexts, env_groups):
        t.set_color(AV_GREY if env == "AV" else ENV[env])
    ax.tick_params(axis="y", length=0)

    yt = ax.get_yaxis_transform()
    env_name = {"Highway": "High-\nspeed", "UrbanFreeway": "Urban\nfreeway",
                "Intersection": "Inter-\nsection", "Roundabout": "Round-\nabout",
                "AV": "AV"}
    LINE_X, TXT_X = -0.235, -0.30
    seen = {}
    for yp, env in zip(y_plot, env_groups):
        seen.setdefault(env, []).append(yp)
    for env, ys in seen.items():
        lo, hi = min(ys), max(ys)
        col = AV_GREY if env == "AV" else ENV[env]
        ax.plot([LINE_X, LINE_X], [lo - bar_h / 2, hi + bar_h / 2],
                transform=yt, color=col, lw=2.4, clip_on=False,
                solid_capstyle="butt")
        ax.text(TXT_X, (lo + hi) / 2, env_name[env], transform=yt,
                rotation=90, va="center", ha="center", fontsize=fs(5.8),
                color=col, fontweight="bold", linespacing=0.85)

    ax.set_xlim(0, 9700)
    ax.set_ylim(-0.8, ymax + 0.8)
    ax.set_xlabel("Curated events per source")
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _: f"{int(v/1000)}k" if v else "0"))
    ax.spines["left"].set_visible(False)

    handles = [Patch(fc=RISK[k], ec="white", label=l) for k, l in
               [("Moderate", "Moderate"), ("High", "High"), ("Extreme", "Extreme")]]
    handles.append(Patch(fc=AV_GREY, ec="white", label="AV (no HV strata)"))
    leg = ax.legend(handles=handles, loc="lower right", ncol=1, handlelength=1.0,
                    handleheight=1.0, labelspacing=0.32, borderpad=0.3,
                    title="Risk level", title_fontsize=fs(6.2),
                    bbox_to_anchor=(1.0, 0.02))
    leg.get_title().set_fontsize(fs(6.2))


# ============================================================================
# Panel b  --  Per-source behaviour-frame breakdown (grouped, log x)
# ============================================================================
def draw_b(ax):
    # md "各源行为帧数明细"
    src_rows = [
        ("HighD",       381254, 247969,  370416),
        ("ExpresswayA",  96867, 107540,    4091),
        ("FreewayB",    164635, 149010, 1101070),
    ]
    behaviours = [("Braking", 1, "#5E86A8"),
                  ("Acceleration", 2, "#86A8C4"),
                  ("Lane change", 3, "#C0763F")]
    n_beh = len(behaviours)
    gap = 1.0
    group_h = n_beh + gap
    ax.set_xscale("log")

    yticks, ylabels = [], []
    for gi, (src, brake, acc, lane) in enumerate(src_rows):
        base = gi * group_h
        vals = {"Braking": brake, "Acceleration": acc, "Lane change": lane}
        for (name, slot, col) in behaviours:
            y = base + (n_beh - slot)
            v = vals[name]
            ax.barh(y, v, height=0.82, color=col, edgecolor="white",
                    linewidth=0.5, zorder=3)
            ax.text(v * 1.12, y, f"{v:,}", va="center", ha="left",
                    fontsize=fs(5.3), color=INK, zorder=4)
        yticks.append(base + (n_beh - 1) / 2.0)
        ylabels.append(src)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=fs(6.4))
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(2e3, 4e6)
    ax.set_ylim(-0.8, (len(src_rows) - 1) * group_h + n_beh - 0.2)
    ax.set_xlabel("Behaviour-labelled frames (log scale)")
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext())

    handles = [Patch(fc=c, ec="white", label=n) for n, _, c in behaviours]
    ax.legend(handles=handles, loc="lower center", ncol=3, handlelength=1.0,
              handleheight=1.0, columnspacing=1.1, handletextpad=0.4,
              borderpad=0.3, bbox_to_anchor=(0.5, 1.0))


# ============================================================================
# Panel c  --  Agent composition (top 100% bar) + non-car class counts (bottom)
# ============================================================================
def draw_c(ax_top, ax_bot):
    TOTAL_USERS = 53295
    strict_vru, heavy_other, car = 4211, 5656, 43428

    # top: 100% composition bar
    segs = [("Passenger car", car, CAR_GREY),
            ("Strict VRU", strict_vru, VRU_WARM),
            ("Heavy / other", heavy_other, HEAVY)]
    left = 0.0
    for name, val, col in segs:
        w = 100 * val / TOTAL_USERS
        ax_top.barh(0, w, left=left, height=0.5, color=col,
                    edgecolor="white", linewidth=0.6)
        if w > 20:
            ax_top.text(left + w / 2, 0, f"Car {w:.1f}%", va="center",
                        ha="center", fontsize=fs(6.0), color=INK)
        left += w
    x_car = 100 * car / TOTAL_USERS
    ax_top.annotate("", xy=(100, 0.5), xytext=(x_car, 0.5),
                    arrowprops=dict(arrowstyle="-", color="#6A6A6A", lw=0.8))
    ax_top.text((x_car + 100) / 2, 0.66, "non-car 18.5%", ha="center",
                va="bottom", fontsize=fs(5.6), color="#6A6A6A")
    ax_top.set_xlim(0, 100)
    ax_top.set_ylim(-0.45, 1.0)
    ax_top.axis("off")

    # bottom: non-car class breakdown (counts)
    classes = [("Motorcycle", 97, VRU_WARM), ("Trailer", 189, HEAVY),
               ("Van", 413, HEAVY), ("Bicycle", 1796, VRU_WARM),
               ("Pedestrian", 2318, VRU_WARM), ("Truck / bus", 5054, HEAVY)]
    yc = np.arange(len(classes))
    for i, (name, val, col) in enumerate(classes):
        ax_bot.barh(yc[i], val, height=0.7, color=col, edgecolor="white",
                    linewidth=0.6)
        ax_bot.text(val + 90, yc[i], f"{val:,}", va="center", ha="left",
                    fontsize=fs(6.0), color=INK)
    ax_bot.set_yticks(yc)
    ax_bot.set_yticklabels([n for n, _, _ in classes], fontsize=fs(6.6))
    ax_bot.tick_params(axis="y", length=0)
    ax_bot.set_xlim(0, 5950)
    ax_bot.set_xlabel("Non-car agents (count)")
    ax_bot.spines["left"].set_visible(False)
    ax_bot.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _: f"{int(v/1000)}k" if v >= 1000 else f"{int(v)}"))
    handles = [Patch(fc=VRU_WARM, ec="white", label="Strict VRU (7.9%)"),
               Patch(fc=HEAVY, ec="white", label="Heavy / special (10.6%)")]
    ax_bot.legend(handles=handles, loc="lower right", handlelength=1.0,
                  handleheight=1.0, labelspacing=0.3, borderpad=0.3,
                  bbox_to_anchor=(1.0, 0.0))


# ============================================================================
# Panel d  --  Speed distribution per scenario (ridgeline, task5 hist)
# ============================================================================
def draw_d(ax):
    spd = _load("task5_speed_distribution_by_dataset_hist.csv")
    spd["center"] = (spd["bin_left_mps"] + spd["bin_right_mps"]) / 2
    summ = _load("task5_speed_distribution_by_dataset_summary.csv")
    medians = dict(zip(summ["dataset"], summ["median"]))

    scen_env = {"HighD": "Highway", "NGSIM": "Highway",
                "ExpresswayA": "UrbanFreeway", "FreewayB": "UrbanFreeway",
                "RounD": "Roundabout", "InD": "Intersection"}
    order = sorted(medians, key=medians.get, reverse=True)

    overlap = 1.65
    for i, scen in enumerate(order):
        sub = spd[spd["dataset"] == scen].sort_values("center")
        x = sub["center"].to_numpy()
        dens = sub["density"].to_numpy()
        dens = dens / dens.max()
        base = (len(order) - 1 - i)
        col = ENV[scen_env[scen]]
        ax.fill_between(x, base, base + dens * overlap, color=col, alpha=0.78,
                        lw=0.0, zorder=10 - i)
        ax.plot(x, base + dens * overlap, color="white", lw=0.6, zorder=10 - i)
        ax.plot(x, base + dens * overlap, color=INK, lw=0.45, alpha=0.55,
                zorder=10 - i)
        m = medians[scen]
        ax.plot([m, m], [base, base + 0.22], color=INK, lw=0.8, zorder=20)
        ax.text(-1.5, base + 0.18, scen, ha="right", va="bottom",
                fontsize=fs(6.2), color=col, fontweight="bold")

    ax.set_xlim(-0.5, 56)
    ax.set_ylim(-0.25, len(order) - 1 + overlap + 0.55)
    ax.set_yticks([])
    ax.set_xlabel("Ego speed (m/s)")
    ax.spines["left"].set_visible(False)


# ============================================================================
# Panel e  --  speed x TTC joint density + marginals (HighD full, task4)
# ============================================================================
SPD_MAX, TTC_MAX = 45.0, 45.0


def _pivot(df):
    df = df.copy()
    df["sc"] = (df["speed_left_mps"] + df["speed_right_mps"]) / 2
    df["tc"] = (df["ttc_left_s"] + df["ttc_right_s"]) / 2
    sc = np.sort(df["sc"].unique())
    tc = np.sort(df["tc"].unique())
    Z = (df.pivot_table(index="tc", columns="sc", values="joint_density",
                        aggfunc="sum").reindex(index=tc, columns=sc)
           .fillna(0.0).to_numpy())
    return sc, tc, Z


def draw_e(ax_top, ax_main, ax_rt, ax_leg):
    joint = _load("task4_ttc_speed_joint_distribution.csv")
    jk = joint[joint["dataset"] == "K-Risk"]
    jo = joint[joint["dataset"] == "Original"]
    sc_k, tc_k, Zk = _pivot(jk)
    sc_o, tc_o, Zo = _pivot(jo)

    levels_k = np.linspace(Zk.max() * 0.04, Zk.max(), 9)
    ax_main.contourf(sc_k, tc_k, Zk, levels=levels_k, cmap=KRISK_CMAP, alpha=0.92)
    levels_o = np.linspace(Zo.max() * 0.10, Zo.max(), 5)
    ax_main.contour(sc_o, tc_o, Zo, levels=levels_o, colors=[ORIG_C],
                    linewidths=0.7, alpha=0.9)
    ax_main.axhline(5.0, color=INK, lw=0.6, ls=(0, (4, 2)), alpha=0.7)
    ax_main.text(44, 5.6, "TTC = 5 s", ha="right", va="bottom", fontsize=fs(5.2),
                 color=INK, alpha=0.8)
    ax_main.set_xlim(0, SPD_MAX)
    ax_main.set_ylim(0, TTC_MAX)
    ax_main.set_xlabel("Ego speed (m/s)")
    ax_main.set_ylabel("Time-to-collision (s)")

    # top marginal: speed (dedicated task4_speed_distribution.csv)
    sp = _load("task4_speed_distribution.csv")
    scen = (sp["bin_left_mps"] + sp["bin_right_mps"]) / 2
    msk = scen <= SPD_MAX
    kd = sp["krisk_density"][msk] / sp["krisk_density"][msk].max()
    od = sp["original_density"][msk] / sp["original_density"][msk].max()
    ax_top.fill_between(scen[msk], 0, kd, color=KRISK_C, alpha=0.30, lw=0)
    ax_top.plot(scen[msk], kd, color=KRISK_C, lw=0.9)
    ax_top.plot(scen[msk], od, color=ORIG_C, lw=0.9)
    ax_top.set_xlim(0, SPD_MAX)
    ax_top.set_ylim(0, 1.18)
    ax_top.axis("off")

    # right marginal: TTC (task4_ttc_distribution.csv)
    ttc = _load("task4_ttc_distribution.csv")
    tcen = (ttc["bin_left_s"] + ttc["bin_right_s"]) / 2
    kdt = ttc["krisk_density"] / ttc["krisk_density"].max()
    odt = ttc["original_density"] / ttc["original_density"].max()
    ax_rt.fill_betweenx(tcen, 0, kdt, color=KRISK_C, alpha=0.30, lw=0)
    ax_rt.plot(kdt, tcen, color=KRISK_C, lw=0.9)
    ax_rt.plot(odt, tcen, color=ORIG_C, lw=0.9)
    ax_rt.set_ylim(0, TTC_MAX)
    ax_rt.set_xlim(0, 1.18)
    ax_rt.axis("off")

    ax_leg.axis("off")
    handles = [Patch(fc=to_rgba(KRISK_C, 0.55), ec=KRISK_C, label="K-Risk"),
               plt.Line2D([0], [0], color=ORIG_C, lw=1.0, label="Original")]
    ax_leg.legend(handles=handles, loc="center", handlelength=1.1,
                  labelspacing=0.4, borderpad=0.2)


# ============================================================================
# Combined figure (manuscript) -- panel letters added ONLY here
# ============================================================================
def panel_label(ax, s, x, y):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left", color=INK)


def build_combined():
    fig = plt.figure(figsize=(7.2, 7.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.92], hspace=0.30)

    top = outer[0].subgridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.46)
    ax_a = fig.add_subplot(top[0, 0])
    # right column: b on top, c (two stacked) below; total matches a height
    right = top[0, 1].subgridspec(2, 1, height_ratios=[0.92, 1.08], hspace=0.55)
    ax_b = fig.add_subplot(right[0, 0])
    c_cell = right[1, 0].subgridspec(2, 1, height_ratios=[0.34, 1.0], hspace=0.32)
    ax_c1 = fig.add_subplot(c_cell[0, 0])
    ax_c2 = fig.add_subplot(c_cell[1, 0])

    bot = outer[1].subgridspec(1, 2, width_ratios=[0.92, 1.18], wspace=0.34)
    ax_d = fig.add_subplot(bot[0, 0])
    e_cell = bot[0, 1].subgridspec(2, 2, width_ratios=[4.0, 1.0],
                                   height_ratios=[1.0, 3.6],
                                   wspace=0.06, hspace=0.06)
    ax_e_top = fig.add_subplot(e_cell[0, 0])
    ax_e = fig.add_subplot(e_cell[1, 0])
    ax_e_rt = fig.add_subplot(e_cell[1, 1])
    ax_e_leg = fig.add_subplot(e_cell[0, 1])

    draw_a(ax_a)
    draw_b(ax_b)
    draw_c(ax_c1, ax_c2)
    draw_d(ax_d)
    draw_e(ax_e_top, ax_e, ax_e_rt, ax_e_leg)

    panel_label(ax_a, "a", x=-0.40, y=1.02)
    panel_label(ax_b, "b", x=-0.34, y=1.06)
    panel_label(ax_c1, "c", x=-0.10, y=1.10)
    panel_label(ax_d, "d", x=-0.07, y=1.02)
    panel_label(ax_e_top, "e", x=-0.16, y=1.02)

    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(OUTDIR, f"figure_overview.{ext}"),
                    dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(HERE, "_preview_overview.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Individual panels (slide deck / paper composition)
#   - NO letters, NO titles, NO grey captions
#   - text enlarged via _FONT_SCALE so in-panel text sits close to (but below)
#     the combined-figure panel-letter size (~10 pt)
# ============================================================================
# Base panel sizes (combined-figure proportions).  Enlarged exports multiply
# these so the bigger text is not cramped.
_PANEL_SIZES = {
    "panel_a_source_risk": (3.5, 3.2),
    "panel_b_behaviour":   (3.4, 2.4),
    "panel_c_agents":      (3.2, 2.9),
    "panel_d_speed_ridgeline": (3.4, 3.0),
    "panel_e_ttc_speed":   (3.8, 3.4),
}


def _draw_panel(name, figsize):
    """Build one panel figure (no letters/titles) and return it."""
    if name == "panel_a_source_risk":
        fig = plt.figure(figsize=figsize); draw_a(fig.add_subplot(111))
    elif name == "panel_b_behaviour":
        fig = plt.figure(figsize=figsize); draw_b(fig.add_subplot(111))
    elif name == "panel_c_agents":
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 1, height_ratios=[0.34, 1.0], hspace=0.32)
        draw_c(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]))
    elif name == "panel_d_speed_ridgeline":
        fig = plt.figure(figsize=figsize); draw_d(fig.add_subplot(111))
    elif name == "panel_e_ttc_speed":
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, width_ratios=[4.0, 1.0],
                              height_ratios=[1.0, 3.6], wspace=0.06, hspace=0.06)
        draw_e(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]),
               fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[0, 1]))
    else:
        raise ValueError(name)
    return fig


def export_individual(outdir=PANELDIR, exts=("png", "pdf"), scale=1.0,
                      size_mul=1.0):
    """Export each panel on its own.

    scale     -> in-panel font scaling (see fs()/_apply_font_scale).
    size_mul  -> multiply panel figsize so enlarged text stays uncramped.
    """
    _apply_font_scale(scale)
    try:
        for name, (w, h) in _PANEL_SIZES.items():
            fig = _draw_panel(name, (w * size_mul, h * size_mul))
            for ext in exts:
                fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=300,
                            bbox_inches="tight")
            plt.close(fig)
    finally:
        _apply_font_scale(1.0)


if __name__ == "__main__":
    build_combined()
    # Original slide-deck panels (unchanged sizes), PNG + PDF -> panels/.
    export_individual()
    # Enlarged paper-composition panels, PDF only -> ImageGen/fig_statistic.
    export_individual(outdir=INDIVDIR, exts=("pdf",), scale=2.0, size_mul=1.88)
    print("combined ->", OUTDIR)
    print("panels   ->", PANELDIR)
    print("enlarged ->", INDIVDIR)
