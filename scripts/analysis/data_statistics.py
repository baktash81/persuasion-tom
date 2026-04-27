"""
Summary statistics and optional figures for the CMV / ToM analysis subset.

Figures are written to figures/ at the repository root.

- If data/posts_with_deltas_clean.json is present (not shipped; very large), the
  script reproduces paper-style stats for the full cleaned corpus (e.g. year counts).
- Otherwise, statistics and plots use only data/cmv_analysis_subset.json (3,040 posts).

Usage (from repository root):
  python scripts/analysis/data_statistics.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
FIG_DIR = REPO_ROOT / "figures"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    subset_path = DATA_DIR / "cmv_analysis_subset.json"
    full_path = DATA_DIR / "posts_with_deltas_clean.json"
    pairs_path = DATA_DIR / "tom_pairs.json"

    print("Loading analysis subset...")
    with open(subset_path, encoding="utf-8") as f:
        subset_raw = json.load(f)
    subset_posts = subset_raw["posts"]

    analysis_ids = None
    if pairs_path.exists():
        with open(pairs_path, encoding="utf-8") as f:
            pairs_data = json.load(f)
        analysis_ids = {p["post_id"] for p in pairs_data["pairs"]}
        subset_for_pairs = [p for p in subset_posts if p["post"]["id"] in analysis_ids]
    else:
        subset_for_pairs = subset_posts

    use_full = full_path.exists()
    if use_full:
        print(f"Loading full clean corpus from {full_path.name}...")
        with open(full_path, encoding="utf-8") as f:
            full_raw = json.load(f)
        posts = full_raw["posts"]
        meta = full_raw.get("metadata", {})
    else:
        posts = subset_posts
        meta = subset_raw.get("metadata", {})
        print("(Full clean JSON not found; using subset for corpus-level plots.)")

    # ─── Console stats ───
    print(f"\n{'=' * 60}")
    print("CORPUS" + (" (full clean)" if use_full else " (subset only)"))
    print(f"{'=' * 60}")
    print(f"Total posts in file: {len(posts):,}")
    if meta:
        print(f"Metadata: {json.dumps(meta, indent=2)}")

    timestamps = [p["post"]["created_utc"] for p in posts]
    dates = [datetime.utcfromtimestamp(t) for t in timestamps]
    print(f"Time range: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}")

    hard_total = sum(len(p.get("hard_negatives", [])) for p in posts)
    easy_total = sum(len(p.get("easy_negatives", [])) for p in posts)
    print(f"Hard negatives (instances): {hard_total:,}")
    print(f"Easy negatives (instances): {easy_total:,}")

    post_len = [len(p["post"].get("selftext", "").split()) for p in posts]
    pers_len = [len(p["persuasive_comment"].get("body", "").split()) for p in posts]
    hard_len = [len(c.get("body", "").split()) for p in posts for c in p.get("hard_negatives", [])]
    easy_len = [len(c.get("body", "").split()) for p in posts for c in p.get("easy_negatives", [])]

    for name, arr in [
        ("Posts", post_len),
        ("Persuasive", pers_len),
        ("Hard neg", hard_len),
        ("Easy neg", easy_len),
    ]:
        a = np.array(arr)
        print(f"{name}: mean={a.mean():.0f}, median={np.median(a):.0f}, std={a.std():.0f}")

    pers_not_top = 0
    total_with_neg = 0
    for p in posts:
        negs = p.get("hard_negatives", []) + p.get("easy_negatives", [])
        if not negs:
            continue
        total_with_neg += 1
        ps = p["persuasive_comment"].get("score", 0)
        if max(c.get("score", 0) for c in negs) > ps:
            pers_not_top += 1
    if total_with_neg:
        print(
            f"\nPosts where some non-delta outscores persuasive: "
            f"{pers_not_top}/{total_with_neg} ({100 * pers_not_top / total_with_neg:.1f}%)"
        )

    print(f"\n{'=' * 60}")
    print(f"ANALYSIS SUBSET ({len(subset_for_pairs):,} posts with complete pairs)")
    print(f"{'=' * 60}")

    sub_pers_len = [len(p["persuasive_comment"].get("body", "").split()) for p in subset_for_pairs]
    sub_hard_len = [len(c.get("body", "").split()) for p in subset_for_pairs for c in p.get("hard_negatives", [])]
    sub_easy_len = [len(c.get("body", "").split()) for p in subset_for_pairs for c in p.get("easy_negatives", [])]
    sub_post_len = [len(p["post"].get("selftext", "").split()) for p in subset_for_pairs]

    for name, arr in [
        ("Posts", sub_post_len),
        ("Persuasive", sub_pers_len),
        ("Hard neg", sub_hard_len),
        ("Easy neg", sub_easy_len),
    ]:
        a = np.array(arr)
        print(f"{name}: n={len(arr):,}, mean={a.mean():.0f}, median={np.median(a):.0f}")

    # ─── Figure: year distribution ───
    years = [datetime.utcfromtimestamp(t).year for t in timestamps]
    year_counts = Counter(years)
    ys = sorted(year_counts.keys())
    cs = [year_counts[y] for y in ys]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(ys, cs, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of posts")
    title_n = f"n={len(posts):,}"
    ax.set_title(f"CMV posts by year ({title_n})" + ("" if use_full else "; released subset file only"))
    ax.set_xticks(ys)
    ax.set_xticklabels(ys, rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "year_distribution.png", dpi=150)
    plt.savefig(FIG_DIR / "year_distribution.pdf", dpi=150)
    plt.close()
    print("\nSaved figures/year_distribution.png")

    # ─── Figure: length distributions (same corpus as `posts`) ───
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    bins = np.arange(0, 801, 20)
    configs = [
        (axes[0, 0], post_len, "Posts", "#4C72B0"),
        (axes[0, 1], pers_len, "Persuasive comments", "#55A868"),
        (axes[1, 0], hard_len, "Hard negative comments", "#C44E52"),
        (axes[1, 1], easy_len, "Easy negative comments", "#8172B2"),
    ]
    for ax, lengths, title, color in configs:
        arr = np.array(lengths)
        ax.hist(arr, bins=bins, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Word count", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_xlim(0, 800)
        ax.axvline(np.median(arr), color="black", linestyle="--", linewidth=1.2, label=f"Median: {int(np.median(arr))}")
        ax.axvline(np.mean(arr), color="black", linestyle=":", linewidth=1.2, label=f"Mean: {int(np.mean(arr))}")
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "length_distribution.png", dpi=150)
    plt.savefig(FIG_DIR / "length_distribution.pdf", dpi=150)
    plt.close()
    print("Saved figures/length_distribution.png")

    # ─── Figure: schematic pipeline (counts match paper when full JSON used) ───
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.axis("off")
    boxes = [
        ("Arctic Shift\nr/ChangeMyView\n2013–2026", 0.5, 0.95, "#E8F0FE"),
        ("Delta detection\n(!delta, Δ)\n40,451 posts (paper)", 0.5, 0.80, "#E8F0FE"),
        ("Remove deleted\n→ 38,064 posts (paper)", 0.5, 0.65, "#FDE8E8"),
        ("Random subset + ToM\n→ analysis file\n3,040 posts", 0.5, 0.50, "#FFF3E0"),
        ("19,340 pairs\n3,040 P + 8,150 H + 8,150 E", 0.5, 0.35, "#E8F0FE"),
    ]
    for text, x, y, color in boxes:
        bbox = dict(boxstyle="round,pad=0.4", facecolor=color, edgecolor="#333333", linewidth=1.2)
        ax.text(x, y, text, ha="center", va="center", fontsize=9, bbox=bbox, family="sans-serif")
    for i in range(len(boxes) - 1):
        ax.annotate(
            "",
            xy=(0.5, boxes[i + 1][2] + 0.055),
            xytext=(0.5, boxes[i][2] - 0.055),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5),
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0.25, 1.0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "data_pipeline.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / "data_pipeline.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved figures/data_pipeline.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
