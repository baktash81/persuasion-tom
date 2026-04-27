"""
Build analysis dataset by joining TOM values with original samples.

For each complete sample (post + persuasive comment + hard/easy negatives all present in TOM output),
produces a flat list of (post_id, comment_id, comment_type, category, post_values, comment_values).

Output: results/tom_pairs.json
  {
    "pairs": [
      {
        "post_id": "abc",
        "comment_id": "xyz",
        "comment_type": "persuasive" | "hard_negative" | "easy_negative",
        "post_values": { "beliefs": "...", "desires": "...", ... },
        "comment_values": { "beliefs": "...", "desires": "...", ... },
        "post_analysis": "...",
        "comment_analysis": "..."
      },
      ...
    ],
    "metadata": { ... }
  }

Usage (from repository root):
  python scripts/analysis/build_dataset.py \
    --input data/cmv_analysis_subset.json \
    --tom data/tom_values.json \
    --output data/tom_pairs.json
"""

import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[2]


TOM_CATEGORIES = ["beliefs", "desires", "intentions", "emotions", "knowledge", "perspective_taking"]


def post_key(post: dict) -> str:
    pid = post.get("id") or post.get("link_id", "").replace("t3_", "")
    if pid:
        return pid
    text = f"{post.get('title', '')}\n{post.get('selftext', '')}"
    return "p_" + hashlib.sha256(text.encode()).hexdigest()[:16]


def comment_key(comment: dict, pid: str) -> str:
    cid = comment.get("id")
    if cid:
        return cid
    text = f"{pid}\n{comment.get('body', '')}"
    return "c_" + hashlib.sha256(text.encode()).hexdigest()[:16]


def extract_values(tom: dict) -> dict:
    """Extract the values string per TOM category. Returns None for missing/null categories."""
    if not tom:
        return {cat: None for cat in TOM_CATEGORIES}
    return {
        cat: (tom.get(cat) or {}).get("values")
        for cat in TOM_CATEGORIES
    }


def extract_content(tom: dict) -> dict:
    """Extract the content string per TOM category."""
    if not tom:
        return {cat: None for cat in TOM_CATEGORIES}
    return {
        cat: (tom.get(cat) or {}).get("content")
        for cat in TOM_CATEGORIES
    }


def build_pairs(samples: list, tom_posts: dict, tom_comments: dict) -> list:
    pairs = []
    skipped = 0

    for sample in samples:
        post = sample.get("post", {})
        pid = post_key(post)
        pc = sample.get("persuasive_comment") or {}
        hard_negs = sample.get("hard_negatives", [])
        easy_negs = sample.get("easy_negatives", [])

        # Post must be in TOM output
        if pid not in tom_posts:
            skipped += 1
            continue

        post_tom = tom_posts[pid].get("tom")
        post_vals = extract_values(post_tom)
        post_content = extract_content(post_tom)
        post_analysis = (post_tom or {}).get("analysis")

        # Build comment entries: (comment_dict, type_label)
        comment_entries = []
        if pc:
            comment_entries.append((pc, "persuasive"))
        for c in hard_negs:
            comment_entries.append((c, "hard_negative"))
        for c in easy_negs:
            comment_entries.append((c, "easy_negative"))

        for comment, ctype in comment_entries:
            cid = comment_key(comment, pid)
            if cid not in tom_comments:
                continue
            c_tom = tom_comments[cid].get("tom")
            c_vals = extract_values(c_tom)
            c_content = extract_content(c_tom)
            c_analysis = (c_tom or {}).get("analysis")

            pairs.append({
                "post_id": pid,
                "comment_id": cid,
                "comment_type": ctype,
                "post_values": post_vals,
                "comment_values": c_vals,
                "post_content": post_content,
                "comment_content": c_content,
                "post_analysis": post_analysis,
                "comment_analysis": c_analysis,
            })

    print(f"Built {len(pairs)} pairs from {len(samples) - skipped} samples ({skipped} skipped - post TOM not yet ready)")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Build TOM pair dataset for analysis")
    parser.add_argument("--input", "-i", type=Path, default=REPO_ROOT / "data" / "cmv_analysis_subset.json")
    parser.add_argument("--tom", "-t", type=Path, default=REPO_ROOT / "data" / "tom_values.json", help="Output of scripts/extraction/run_tom_extraction.py")
    parser.add_argument("--output", "-o", type=Path, default=REPO_ROOT / "data" / "tom_pairs.json")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    with open(args.input, encoding="utf-8") as f:
        raw = json.load(f)
    samples = raw.get("posts") or raw.get("samples") or []
    print(f"  {len(samples)} total samples")

    print(f"Loading {args.tom}...")
    with open(args.tom, encoding="utf-8") as f:
        tom_data = json.load(f)
    tom_posts = tom_data.get("posts", {})
    tom_comments = tom_data.get("comments", {})
    print(f"  {len(tom_posts)} posts, {len(tom_comments)} comments with TOM values")

    pairs = build_pairs(samples, tom_posts, tom_comments)

    # Count by type
    type_counts = {}
    for p in pairs:
        t = p["comment_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"Pair breakdown: {type_counts}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "metadata": {
            "built_at": datetime.now().isoformat(),
            "source_input": str(args.input),
            "source_tom": str(args.tom),
            "total_pairs": len(pairs),
            "type_counts": type_counts,
        },
        "pairs": pairs,
    }
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
