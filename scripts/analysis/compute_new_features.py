"""
Compute two new ROSCOE-inspired feature families:

1. FAITHFULNESS (Grounding Score):
   For each TOM category, how well are the extracted value keywords
   grounded in the actual comment text? We compute cosine similarity
   between the mean-pooled value-keyword embedding and the sentence
   embedding of the comment's content description for that category.

2. INTERNAL CONSISTENCY:
   How coherent are the TOM categories within a single comment?
   We compute pairwise cosine similarity between all 6 category
   mean-pooled value embeddings and report the average as a
   consistency score. We also compute per-pair consistency
   (e.g., beliefs<->intentions, emotions<->perspective_taking).

Output: results/new_features.csv  (merged with existing tom_features.csv)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from scipy import stats
from itertools import combinations

# ── Config ──────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ['beliefs', 'desires', 'intentions', 'emotions', 'knowledge', 'perspective_taking']
PAIRS_PATH = DATA_DIR / "tom_pairs.json"

# Category pairs for internal consistency breakdown
COGNITIVE_CATS = ['beliefs', 'desires', 'intentions', 'knowledge']
AFFECTIVE_CATS = ['emotions', 'perspective_taking']

# ── Load data ───────────────────────────────────────────────────
print("Loading data...")
with open(PAIRS_PATH, encoding="utf-8") as f:
    data = json.load(f)
pairs = data['pairs']
print(f"Loaded {len(pairs)} pairs")

# ── Load model ──────────────────────────────────────────────────
print("Loading sentence-transformers model...")
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
print("Model loaded.")

# ── Helper functions ────────────────────────────────────────────
def parse_values(val_str):
    """Split comma-separated value string into lowercase tokens."""
    if not val_str:
        return []
    return [v.strip().lower() for v in val_str.split(',') if v.strip()]


# ── Step 1: Embed all unique value tokens ───────────────────────
print("Collecting and embedding value tokens...")
all_tokens = set()
for pair in pairs:
    for side in ['post_values', 'comment_values']:
        for cat in CATEGORIES:
            all_tokens.update(parse_values(pair[side].get(cat)))

all_tokens = list(all_tokens)
print(f"  Unique tokens: {len(all_tokens)}")
token_embs = model.encode(all_tokens, batch_size=512, show_progress_bar=True, normalize_embeddings=True)
tok_map = {t: token_embs[i] for i, t in enumerate(all_tokens)}

# ── Step 2: Embed all unique content descriptions ───────────────
print("Collecting and embedding content descriptions...")
all_contents = set()
for pair in pairs:
    for side in ['post_content', 'comment_content']:
        if pair.get(side):
            for cat in CATEGORIES:
                text = pair[side].get(cat, '')
                if text and text.strip():
                    all_contents.add(text.strip())

# Also embed full analysis strings
for pair in pairs:
    for key in ['post_analysis', 'comment_analysis']:
        text = pair.get(key, '')
        if text and text.strip():
            all_contents.add(text.strip())

all_contents = list(all_contents)
print(f"  Unique content strings: {len(all_contents)}")
content_embs = model.encode(all_contents, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
content_map = {t: content_embs[i] for i, t in enumerate(all_contents)}


def mean_pool(tokens):
    """Mean-pool token embeddings into a unit-normalized vector."""
    vecs = [tok_map[t] for t in tokens if t in tok_map]
    if not vecs:
        return None
    agg = np.mean(vecs, axis=0)
    n = np.linalg.norm(agg)
    return agg / n if n > 0 else agg


def get_content_emb(pair, side, cat):
    """Get the sentence embedding of a content description."""
    text = pair.get(side, {})
    if not text:
        return None
    desc = text.get(cat, '')
    if not desc or not desc.strip():
        return None
    return content_map.get(desc.strip())


# ── Step 3: Compute features ───────────────────────────────────
print("\nComputing faithfulness and internal consistency features...")
rows = []

for i, pair in enumerate(pairs):
    if i % 5000 == 0:
        print(f"  Processing pair {i}/{len(pairs)}...")

    row = {
        'post_id': pair['post_id'],
        'comment_id': pair['comment_id'],
        'comment_type': pair['comment_type'],
    }

    # ── FAITHFULNESS: value keywords vs comment content text ────
    # For each category, compute cosine(mean_pool(value_keywords), sentence_emb(content_description))
    comment_faith_scores = []
    post_faith_scores = []

    for cat in CATEGORIES:
        # Comment faithfulness: are the comment's value keywords grounded in the comment's content?
        comment_val_vec = mean_pool(parse_values(pair['comment_values'].get(cat)))
        comment_content_vec = get_content_emb(pair, 'comment_content', cat)

        if comment_val_vec is not None and comment_content_vec is not None:
            faith = float(np.dot(comment_val_vec, comment_content_vec))
            row[f'{cat}_comment_faith'] = faith
            comment_faith_scores.append(faith)
        else:
            row[f'{cat}_comment_faith'] = np.nan

        # Post faithfulness: are the post's value keywords grounded in the post's content?
        post_val_vec = mean_pool(parse_values(pair['post_values'].get(cat)))
        post_content_vec = get_content_emb(pair, 'post_content', cat)

        if post_val_vec is not None and post_content_vec is not None:
            faith = float(np.dot(post_val_vec, post_content_vec))
            row[f'{cat}_post_faith'] = faith
            post_faith_scores.append(faith)
        else:
            row[f'{cat}_post_faith'] = np.nan

    # Aggregate faithfulness scores
    row['comment_faith_mean'] = np.mean(comment_faith_scores) if comment_faith_scores else np.nan
    row['post_faith_mean'] = np.mean(post_faith_scores) if post_faith_scores else np.nan

    # ── INTERNAL CONSISTENCY: cross-category coherence ──────────
    # Build per-category mean-pooled vectors for the comment
    comment_cat_vecs = {}
    for cat in CATEGORIES:
        vec = mean_pool(parse_values(pair['comment_values'].get(cat)))
        if vec is not None:
            comment_cat_vecs[cat] = vec

    # All pairwise cosine similarities between categories
    cat_pairs_sims = []
    for cat_a, cat_b in combinations(CATEGORIES, 2):
        if cat_a in comment_cat_vecs and cat_b in comment_cat_vecs:
            sim = float(np.dot(comment_cat_vecs[cat_a], comment_cat_vecs[cat_b]))
            row[f'consistency_{cat_a}_{cat_b}'] = sim
            cat_pairs_sims.append(sim)
        else:
            row[f'consistency_{cat_a}_{cat_b}'] = np.nan

    # Overall internal consistency (mean of all pairwise)
    row['comment_consistency_mean'] = np.mean(cat_pairs_sims) if cat_pairs_sims else np.nan

    # Cognitive sub-consistency (beliefs, desires, intentions, knowledge)
    cog_sims = []
    for cat_a, cat_b in combinations(COGNITIVE_CATS, 2):
        if cat_a in comment_cat_vecs and cat_b in comment_cat_vecs:
            cog_sims.append(float(np.dot(comment_cat_vecs[cat_a], comment_cat_vecs[cat_b])))
    row['comment_cognitive_consistency'] = np.mean(cog_sims) if cog_sims else np.nan

    # Affective sub-consistency (emotions, perspective_taking)
    if 'emotions' in comment_cat_vecs and 'perspective_taking' in comment_cat_vecs:
        row['comment_affective_consistency'] = float(
            np.dot(comment_cat_vecs['emotions'], comment_cat_vecs['perspective_taking'])
        )
    else:
        row['comment_affective_consistency'] = np.nan

    # Cross-domain consistency (cognitive vs affective mean)
    cross_sims = []
    for cog_cat in COGNITIVE_CATS:
        for aff_cat in AFFECTIVE_CATS:
            if cog_cat in comment_cat_vecs and aff_cat in comment_cat_vecs:
                cross_sims.append(float(np.dot(comment_cat_vecs[cog_cat], comment_cat_vecs[aff_cat])))
    row['comment_cross_consistency'] = np.mean(cross_sims) if cross_sims else np.nan

    # ── Same for POST internal consistency ──────────────────────
    post_cat_vecs = {}
    for cat in CATEGORIES:
        vec = mean_pool(parse_values(pair['post_values'].get(cat)))
        if vec is not None:
            post_cat_vecs[cat] = vec

    post_pairs_sims = []
    for cat_a, cat_b in combinations(CATEGORIES, 2):
        if cat_a in post_cat_vecs and cat_b in post_cat_vecs:
            post_pairs_sims.append(float(np.dot(post_cat_vecs[cat_a], post_cat_vecs[cat_b])))
    row['post_consistency_mean'] = np.mean(post_pairs_sims) if post_pairs_sims else np.nan

    # ── CONSISTENCY DIFFERENCE: comment - post ──────────────────
    # Does the comment have a more/less coherent TOM profile than the post?
    if not np.isnan(row['comment_consistency_mean']) and not np.isnan(row['post_consistency_mean']):
        row['consistency_diff'] = row['comment_consistency_mean'] - row['post_consistency_mean']
    else:
        row['consistency_diff'] = np.nan

    rows.append(row)

df_new = pd.DataFrame(rows)
print(f"\nNew features computed: {df_new.shape}")

# ── Step 4: Identify feature columns ───────────────────────────
faith_cols = [f'{c}_comment_faith' for c in CATEGORIES] + [f'{c}_post_faith' for c in CATEGORIES] + \
             ['comment_faith_mean', 'post_faith_mean']

consistency_pair_cols = [f'consistency_{a}_{b}' for a, b in combinations(CATEGORIES, 2)]
consistency_agg_cols = ['comment_consistency_mean', 'comment_cognitive_consistency',
                        'comment_affective_consistency', 'comment_cross_consistency',
                        'post_consistency_mean', 'consistency_diff']

all_new_cols = faith_cols + consistency_pair_cols + consistency_agg_cols
print(f"\nTotal new features: {len(all_new_cols)}")
print(f"  Faithfulness: {len(faith_cols)}")
print(f"  Consistency (pairwise): {len(consistency_pair_cols)}")
print(f"  Consistency (aggregate): {len(consistency_agg_cols)}")

# ── Step 5: Statistical analysis ───────────────────────────────
print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS OF NEW FEATURES")
print("=" * 80)

persuasive = df_new[df_new['comment_type'] == 'persuasive']
hard_neg = df_new[df_new['comment_type'] == 'hard_negative']
easy_neg = df_new[df_new['comment_type'] == 'easy_negative']


def stat_test(col, a, b):
    av, bv = a[col].dropna().values, b[col].dropna().values
    if len(av) < 10 or len(bv) < 10:
        return None
    _, p = stats.mannwhitneyu(av, bv, alternative='two-sided')
    pooled_std = np.sqrt((av.std() ** 2 + bv.std() ** 2) / 2)
    d = (av.mean() - bv.mean()) / pooled_std if pooled_std > 0 else 0
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    return {'feature': col, 'mean_a': av.mean(), 'mean_b': bv.mean(),
            'cohen_d': d, 'p_value': p, 'sig': sig}


for label, a, b in [('PERSUASIVE vs HARD NEGATIVE', persuasive, hard_neg),
                     ('PERSUASIVE vs EASY NEGATIVE', persuasive, easy_neg)]:
    print(f"\n{'=' * 80}")
    print(f"{label}")
    print(f"{'=' * 80}")
    print(f"{'Feature':<40} {'mean_P':>8} {'mean_X':>8} {'Cohen d':>9} {'p-value':>12} {'sig':>5}")
    print('-' * 82)

    results_list = []
    for col in all_new_cols:
        r = stat_test(col, a, b)
        if r:
            results_list.append(r)
            print(f"{r['feature']:<40} {r['mean_a']:>8.4f} {r['mean_b']:>8.4f} "
                  f"{r['cohen_d']:>9.4f} {r['p_value']:>12.4e} {r['sig']:>5}")

    results_df = pd.DataFrame(results_list)
    suffix = 'pvh' if 'HARD' in label else 'pve'
    results_df.to_csv(RESULTS_DIR / f'new_features_stats_{suffix}.csv', index=False)

# ── Step 6: Summary tables ──────────────────────────────────────
print("\n\n" + "=" * 80)
print("FAITHFULNESS SUMMARY (mean by comment type)")
print("=" * 80)
faith_summary = df_new.groupby('comment_type')[faith_cols].mean().T
print(faith_summary[['persuasive', 'hard_negative', 'easy_negative']].round(4).to_string())

print("\n\n" + "=" * 80)
print("INTERNAL CONSISTENCY SUMMARY (mean by comment type)")
print("=" * 80)
cons_summary = df_new.groupby('comment_type')[consistency_agg_cols].mean().T
print(cons_summary[['persuasive', 'hard_negative', 'easy_negative']].round(4).to_string())

# ── Step 7: Save ────────────────────────────────────────────────
# Merge with existing features
existing = pd.read_csv(DATA_DIR / "tom_features.csv")
merged = existing.merge(
    df_new[['post_id', 'comment_id'] + all_new_cols],
    on=['post_id', 'comment_id'],
    how='left'
)
merged.to_csv(RESULTS_DIR / 'tom_features_extended.csv', index=False)
print(f"\nSaved extended features: {merged.shape[0]} rows x {merged.shape[1]} cols")
print(f"  -> results/tom_features_extended.csv")

# Also save just the new features
df_new.to_csv(RESULTS_DIR / 'new_features.csv', index=False)
print(f"  -> results/new_features.csv")

print("\nDone!")
