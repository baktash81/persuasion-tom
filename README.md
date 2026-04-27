# Patterns of Persuasion Through the Lens of Theory of Mind

Repository for the **SocialLLM Workshop @ ICWSM 2026** paper: *Patterns of Persuasion Through the Lens of Theory of Mind: Value Alignment Analysis in Online Deliberation*.


## Contents

```
├── data/
│   ├── cmv_analysis_subset.json   # 3,040 CMV posts + persuasive / hard / easy comments
│   ├── tom_pairs.json             # 19,340 pairs with ToM value strings (+ content fields)
│   └── tom_features.csv           # Precomputed alignment / feature table (one row per pair)
├── scripts/
│   ├── data/                      # Arctic Shift helpers + cleaning (optional full pipeline)
│   ├── extraction/                # ToM prompts + vLLM extraction driver
│   └── analysis/                  # build pairs, extended features, stats, notebook
├── results/                       # Default output dir for regenerated tables (gitignored)
├── figures/                       # Written by data_statistics.py
├── DATASHEET.md
├── FAIR.md
├── LICENSE
└── requirements.txt
```

## Data summary

| File | Description |
|------|-------------|
| `cmv_analysis_subset.json` | Analysis posts with `post`, `persuasive_comment`, `hard_negatives`, `easy_negatives`, `delta_reply`. |
| `tom_pairs.json` | Flat list of pairs: `post_id`, `comment_id`, `comment_type`, `post_values`, `comment_values`, `post_content`, `comment_content`, analyses. |
| `tom_features.csv` | Features used for models/stats (similarity, Jaccard, novelty, coverage, counts, `global_sim`, etc.). |

## Quick start (analysis only)

From the repository root:

```bash
pip install -r requirements.txt
# Optional: summary stats + figures (subset-based if full clean JSON absent)
python scripts/analysis/data_statistics.py
```

Explore or regenerate feature-based analyses:

```bash
jupyter notebook scripts/analysis/tom_analysis.ipynb
```

Optional: recompute faithfulness / internal-consistency columns merged into an extended CSV:

```bash
python scripts/analysis/compute_new_features.py
# writes results/tom_features_extended.csv (reads data/tom_features.csv + data/tom_pairs.json)
```

## Full replication path (GPU)

1. **Collect / clean** full CMV corpus (optional; large): see `scripts/data/` and paper Data section. Place `posts_with_deltas_clean.json` under `data/` if you want paper-scale year/length figures from `data_statistics.py`.

2. **ToM extraction** with vLLM + Qwen (two-call strategy per text):

```bash
python scripts/extraction/run_tom_extraction.py \
  --input data/cmv_analysis_subset.json \
  --output data/tom_values.json \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --tensor-parallel-size 3
```

3. **Build pairs:**

```bash
python scripts/analysis/build_dataset.py \
  --input data/cmv_analysis_subset.json \
  --tom data/tom_values.json \
  --output data/tom_pairs.json
```

4. Re-run feature notebook or your own analysis on the new `tom_pairs.json`.

Run extraction and `build_dataset.py` from the **repository root** so default paths resolve.

## Documentation for checklists

- **`DATASHEET.md`** — dataset motivation, composition, limitations, licensing.  
- **`FAIR.md`** — short FAIR-oriented documentation for findability and reuse.

## Citation

If you use this repository, cite the workshop paper (bib entry in the camera-ready PDF) and link to https://github.com/baktash81/persuasion-tom.

## Authors

Baktash Ansari (University of Washington Bothell), Mouly Dewan (UW Seattle), Vibhor Agarwal (Nokia Bell Labs, UK), Afra Mashhadi (UW Bothell).  
**Corresponding:** baktash@uw.edu

## License

- **Code** (`scripts/`): MIT  
- **Data** (`data/`): CC BY-NC 4.0  

See `LICENSE`.
