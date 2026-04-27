# Datasheet: CMV ToM persuasion analysis release

This document follows the spirit of [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) (Gebru et al.) for the artifacts in `data/`.

## Motivation

1. **For what purpose was the dataset created?**  
   To study population-level alignment between Theory-of-Mind (ToM) value profiles of r/ChangeMyView (CMV) posts and replies, contrasting persuasive (delta-awarded) comments with highly engaged but non-persuasive and low-engagement comments.

2. **Who created it and on whose behalf?**  
   University of Washington (Bothell & Seattle) and Nokia Bell Labs researchers; see the paper author list.

3. **What are recommended uses?**  
   Replication of reported statistics, secondary analysis of ToM alignment features, and development of cognitively grounded NLP metrics—not individual-level persuasion prediction.

4. **What are non-recommended uses?**  
   Harassment, deanonymization of Reddit users, commercial reuse of text without regard to Reddit’s terms and CC BY-NC, or deployment as a high-stakes persuasion scoring tool (within-post discrimination is near chance in our study).

## Composition

5. **What do the instances represent?**  
   Public CMV posts and top-level comments with Reddit metadata (e.g. `id`, `score`, `created_utc`, text fields). ToM extractions are LLM-generated structured tags per six categories.

6. **How many instances?**  
   - `cmv_analysis_subset.json`: 3,040 posts with persuasive, hard-negative, and easy-negative comments.  
   - `tom_pairs.json`: 19,340 post–comment pairs (3,040 persuasive + 8,150 hard negative + 8,150 easy negative) with paired ToM value strings and per-category content fields used for analysis.  
   - `tom_features.csv`: one row per pair with precomputed alignment features.

7. **Does the dataset contain sensitive or offensive content?**  
   Text is user-generated public discourse and may include controversial opinions or coarse language. No private messages are included.

8. **Is there labeling?**  
   Persuasive comments are defined by explicit delta awards in the thread (a community/OP signal), not by third-party annotators. **Human annotation labels from our face-validity study are not part of this release.**

## Collection process

9. **How was the data collected?**  
   Historical CMV posts and comments were obtained with the Arctic Shift Reddit archive tooling; posts were filtered (e.g. removal of deleted bodies), delta-awarded persuasive comments identified via markers, and negatives sampled by score quartiles as described in the paper.

10. **Who was involved?**  
    Automated collection and filtering; LLM-based ToM extraction (Qwen3-30B family via vLLM); embedding model `all-MiniLM-L6-v2` for feature computation.

## Preprocessing / cleaning

11. **What preprocessing was applied?**  
    Cleaning removed deleted or removed posts/comments; multi-delta handling kept one persuasive comment per post; non-persuasive comments were constrained to timestamps before the persuasive comment where applicable (see paper Data section).

## Uses

12. **Splits / tasks**  
    The release is organized for three-way comparison at the pair level (`comment_type`: `persuasive`, `hard_negative`, `easy_negative`). Train/validation splits for the logistic model in the paper used grouped cross-validation by `post_id`.

## Maintenance

13. **Version**  
    Bundled metadata in JSON `metadata` objects records build timestamps where applicable. This repository snapshot corresponds to the ICWSM 2026 SocialLLM workshop camera-ready materials.

14. **License**  
    Data: **CC BY-NC 4.0** (see `LICENSE`). Code: **MIT**.

15. **Contact**  
    Corresponding author: **Baktash Ansari** (`baktash@uw.edu`).
