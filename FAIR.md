# FAIR overview for this release

This file supports ethics / reproducibility checklists asking how the dataset supports [FAIR](https://www.go-fair.org/fair-principles/) principles. It summarizes how we address **Findable, Accessible, Interoperable, and Reusable** practices for the artifacts in this repository.

## Findable

- **Persistent public location:** https://github.com/baktash81/persuasion-tom
- **Citation:** Cite the workshop paper (ICWSM 2026 SocialLLM) and this repository DOI/URL once assigned.
- **Identity:** Each post and comment retains Reddit-style `id` fields where available; pair files repeat `post_id` and `comment_id` for joins.

## Accessible

- **Access protocol:** Standard `git clone` / release download; large JSON files are stored in-repo (not behind a paywall).
- **Format:** UTF-8 JSON and CSV; readable with standard tools.
- **Licensing:** Data are released under **CC BY-NC 4.0**; code under **MIT** (see `LICENSE`). Reddit’s own terms apply to underlying content.

## Interoperable

- **Formats:** JSON for nested records; CSV for flat feature tables with explicit column names (`beliefs_sim`, `comment_type`, etc.).
- **External models:** ToM extraction is documented to use **Qwen3-30B-A3B**-class weights via **vLLM**; embeddings use **sentence-transformers** `all-MiniLM-L6-v2`, declared in `requirements.txt` and the paper.

## Reusable

- **Documentation:** `README.md` describes the pipeline; `DATASHEET.md` summarizes composition and limitations; prompts mirror the paper appendix in `scripts/extraction/tom_prompts.py`.
- **Provenance:** `tom_pairs.json` metadata includes `source_input` / `source_tom` paths when rebuilt via `build_dataset.py`.
- **Scope:** Results are **population-level patterns**; within-post prediction is weak by design—documented in the paper and `DATASHEET.md`.

