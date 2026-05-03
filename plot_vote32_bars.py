import os, json
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = "safe_test"
OUT_DIR = "vote32_plots"
os.makedirs(OUT_DIR, exist_ok=True)

BASES = ["Llama-3-8B-Instruct", "Llama-3.2-3B-Instruct", "Qwen3-4B", "Qwen3-8B"]
BENCHES = ["direct", "harm", "harmful_behaviors", "phi"]
SUFFIX = "vector-mode2-top50-horizon{H}-alpha1.0-keep{H}-vote-samples8-T1.0-Tp1.0-Vk200-excl1-PKU_UnSafeRLHF_100"


def unsafe_rate(model_dir, bench):
    """Return (rate_temp0_pct, rate_temp1_avg_pct)."""
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
    r0 = np.mean([rate_one(f) for f in t0_files]) * 100 if t0_files else 0
    r1 = np.mean([rate_one(f) for f in t1_files]) * 100 if t1_files else 0
    return r0, r1


def plot_base(base):
    variant_dirs = [base] + [f"{base}-{SUFFIX.format(H=h)}" for h in range(1, 9)]
    variant_labels = ["base"] + [f"h{h}" for h in range(1, 9)]

    rates_t0 = np.zeros((len(BENCHES), len(variant_dirs)))
    rates_t1 = np.zeros((len(BENCHES), len(variant_dirs)))
    for i, bench in enumerate(BENCHES):
        for j, vd in enumerate(variant_dirs):
            rates_t0[i, j], rates_t1[i, j] = unsafe_rate(vd, bench)

    x = np.arange(len(variant_dirs))
    bar_w = 0.2
    colors = ["#4C72B0", "#DD8452", "#55A467", "#C44E52"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, rates, title in [(axes[0], rates_t0, "Temperature = 0"),
                              (axes[1], rates_t1, "Temperature = 1 (avg of runs)")]:
        for i, bench in enumerate(BENCHES):
            ax.bar(x + (i - 1.5) * bar_w, rates[i], bar_w, label=bench, color=colors[i])
        ax.set_ylabel("Unsafe rate (%)")
        ax.set_title(title)
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(10, rates.max() * 1.15))

    axes[0].legend(ncol=4, loc="upper right")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(variant_labels)
    axes[-1].set_xlabel("Model variant (base + horizon{N}-keep{N})")

    fig.suptitle(f"{base} — vote-samples8 horizon sweep vs base", fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"vote32_{base}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")


for b in BASES:
    plot_base(b)
