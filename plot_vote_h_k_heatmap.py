import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = "safe_test"
OUT_DIR = "vote_h_k_plots"
os.makedirs(OUT_DIR, exist_ok=True)

BASES = ["Llama-3.2-3B-Instruct", "Qwen3-4B"]
BENCHES = ["direct", "harm", "harmful_behaviors", "phi"]

MODEL_PATTERN = re.compile(
    r'^(?P<base>Llama-3\.2-3B-Instruct|Qwen3-4B)'
    r'-vector-mode2-top50'
    r'-horizon(?P<horizon>\d+)'
    r'-alpha1\.0'
    r'-keep(?P<keep>\d+)'
    r'-vote-samples8-T1\.0-Tp1\.0-Vk200-excl1-PKU_UnSafeRLHF_100$'
)


def unsafe_rate(model_dir, bench):
    eval_dir = os.path.join(BASE_DIR, model_dir, "eval_results")
    if not os.path.isdir(eval_dir):
        return np.nan, np.nan

    def rate_one(fp):
        d = json.load(open(fp))
        if not d:
            return np.nan
        return sum(1 for x in d if x.get("score") != "safe") / len(d)

    t0_files, t1_files = [], []
    for sub in os.listdir(eval_dir):
        sub_path = os.path.join(eval_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        for f in os.listdir(sub_path):
            if not f.endswith(f"-{bench}.json"):
                continue
            (t0_files if sub == "temp_0" else t1_files).append(os.path.join(sub_path, f))
    r0 = np.mean([rate_one(f) for f in t0_files]) * 100 if t0_files else np.nan
    r1 = np.mean([rate_one(f) for f in t1_files]) * 100 if t1_files else np.nan
    return r0, r1


def discover(base):
    items = []
    for name in os.listdir(BASE_DIR):
        m = MODEL_PATTERN.match(name)
        if not m or m.group('base') != base:
            continue
        items.append({
            'model': name,
            'horizon': int(m.group('horizon')),
            'keep': int(m.group('keep')),
        })
    return items


def make_grid(items, bench):
    horizons = sorted({it['horizon'] for it in items})
    keeps = sorted({it['keep'] for it in items})
    h_idx = {h: i for i, h in enumerate(horizons)}
    k_idx = {k: i for i, k in enumerate(keeps)}
    g0 = np.full((len(horizons), len(keeps)), np.nan)
    g1 = np.full((len(horizons), len(keeps)), np.nan)
    for it in items:
        r0, r1 = unsafe_rate(it['model'], bench)
        g0[h_idx[it['horizon']], k_idx[it['keep']]] = r0
        g1[h_idx[it['horizon']], k_idx[it['keep']]] = r1
    return horizons, keeps, g0, g1


def annotate(ax, grid, fmt="{:.1f}"):
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isnan(v):
                continue
            color = "white" if v > grid[~np.isnan(grid)].max() * 0.55 else "black"
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=8, color=color)


def plot_base(base):
    items = discover(base)
    if not items:
        print(f"no data for {base}")
        return

    fig, axes = plt.subplots(2, len(BENCHES), figsize=(4.6 * len(BENCHES), 8.5),
                             squeeze=False)
    horizons_full = None
    keeps_full = None

    for col, bench in enumerate(BENCHES):
        horizons, keeps, g0, g1 = make_grid(items, bench)
        horizons_full, keeps_full = horizons, keeps

        vmax = np.nanmax([g0, g1])
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0

        for row, (g, label) in enumerate([(g0, "T=0"), (g1, "T=1 avg")]):
            ax = axes[row, col]
            im = ax.imshow(g, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(keeps)))
            ax.set_xticklabels([f"k{k}" for k in keeps], fontsize=9)
            ax.set_yticks(range(len(horizons)))
            ax.set_yticklabels([f"h{h}" for h in horizons], fontsize=9)
            ax.set_title(f"{bench} — {label}", fontsize=10)
            if col == 0:
                ax.set_ylabel("horizon")
            if row == 1:
                ax.set_xlabel("keep")
            annotate(ax, g)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="unsafe %")

    fig.suptitle(f"{base} — unsafe rate by (horizon, keep), vote-samples8",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT_DIR, f"vote_h_k_{base}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}  ({len(items)} variants)")


for b in BASES:
    plot_base(b)
