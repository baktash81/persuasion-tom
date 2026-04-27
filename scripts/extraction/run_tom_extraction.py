#!/usr/bin/env python3
"""
Extract TOM (Theory of Mind) values from posts and comments for delta_samples.json
or posts_with_deltas_clean.json.

Uses TOM prompts to analyze posts and comments across 6 TOM categories:
- Beliefs, Desires, Intentions, Emotions, Knowledge, Perspective-taking

Supports two input formats:
- "samples": list under key "samples", each with post, persuasive_comment, comments
- "posts": list under key "posts", each with post, persuasive_comment, hard_negatives, easy_negatives
  (comment TOM is run on persuasive_comment + hard_negatives + easy_negatives)

Writes TOM results to a separate output file keyed by unique post/comment ids (or text hash if no id).
Input file is never modified. Output format: { "metadata", "posts": { "<id>": { "id", "tom", ... } }, "comments": { "<id>": { "id", "post_id", "tom", ... } } }.
"""
# Disable vLLM/HuggingFace internal progress bars so only our main tqdm bar is visible
import os
os.environ["VLLM_DISABLE_TQDM"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import hashlib
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Optional

from tqdm import tqdm

from vllm_inference import QwenInference as BaseQwenInference
from tom_prompts import (
    TOM_POST_PROMPT_PART1,
    TOM_POST_PROMPT_PART2,
    TOM_COMMENT_PROMPT_PART1,
    TOM_COMMENT_PROMPT_PART2,
    build_tom_post_prompt,
    build_tom_comment_prompt,
    parse_tom_response,
    merge_parsed_tom,
)


def _post_key(post: dict) -> str:
    """Unique key for a post: post['id'] if present, else hash of title + selftext."""
    pid = post.get("id") or post.get("link_id", "").replace("t3_", "")
    if pid:
        return pid
    text = f"{post.get('title', '')}\n{post.get('selftext', '')}"
    return "p_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _comment_key(comment: dict, post_id: str) -> str:
    """Unique key for a comment: comment['id'] if present, else hash of post_id + body."""
    cid = comment.get("id")
    if cid:
        return cid
    text = f"{post_id}\n{comment.get('body', '')}"
    return "c_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _get_comment_targets(sample: dict) -> list[tuple[dict, dict]]:
    """
    Return list of (target_dict, comment_dict) for comment-level TOM.
    Supports both formats: samples (comments list) and posts (persuasive + hard + easy).
    """
    targets = []
    pc = sample.get("persuasive_comment") or {}
    if pc:
        targets.append((pc, pc))
    # Legacy format: single "comments" list
    for c in sample.get("comments", []):
        targets.append((c, c))
    # posts_with_deltas_clean format: hard_negatives + easy_negatives
    for c in sample.get("hard_negatives", []):
        targets.append((c, c))
    for c in sample.get("easy_negatives", []):
        targets.append((c, c))
    return targets


def run_inference_on_delta_samples(
    input_path: str,
    output_path: Optional[str] = None,
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    device: str = "auto",
    torch_dtype: str = "auto",
    max_new_tokens: int = 8000,
    limit: Optional[int] = None,
    enable_thinking: bool = True,
    tensor_parallel_size: int = 1,
    max_model_len: int = 12000,
    gpu_memory_utilization: float = 0.9,
    cpu_offload_gb: float = 0,
    enforce_eager: bool = False,
    random_sample: Optional[int] = None,
    seed: Optional[int] = None,
):
    """
    Process input JSON: extract TOM values from posts and comments for each sample.
    Supports "samples" or "posts" key. Comment TOM runs on persuasive + hard_negatives + easy_negatives.
    Writes only TOM results to a separate output file, keyed by post id and comment id (or text hash if no id).
    Input file is never modified. Resume: existing output file is loaded and already-done keys are skipped.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_tom_values{input_path.suffix}"
    else:
        output_path = Path(output_path)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples") or data.get("posts") or []
    total_available = len(samples)
    if random_sample is not None:
        if seed is not None:
            random.seed(seed)
        n = min(random_sample, total_available)
        samples = random.sample(samples, n)
        print(f"Random sample: {n} of {total_available} total")
    elif limit:
        samples = samples[:limit]

    # Load existing output for resume (only posts/comments dicts; keyed by unique id)
    out_posts: dict = {}
    out_comments: dict = {}
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            raw_posts = existing.get("posts")
            raw_comments = existing.get("comments")
            # New format uses dicts keyed by id; old format had lists — only use if dict
            if isinstance(raw_posts, dict):
                out_posts = raw_posts
            if isinstance(raw_comments, dict):
                out_comments = raw_comments
            if out_posts or out_comments:
                print(f"Resuming: loaded {len(out_posts)} posts, {len(out_comments)} comments from {output_path}")
        except Exception as e:
            print(f"Could not load existing output ({e}); starting fresh.")

    total_inferences = 0
    for sample in samples:
        n = 1 + len(_get_comment_targets(sample))  # 1 post + N comments
        total_inferences += n * 2  # 2 runs per item (part1 + part2)

    print(f"TOM mode: two-run (part1: beliefs+desires+intentions, part2: emotions+knowledge+perspective_taking)")
    print(f"Loading model: {model_name} (tensor_parallel_size={tensor_parallel_size})")
    vllm_kwargs = {}
    vllm_kwargs["max_model_len"] = max_model_len
    if gpu_memory_utilization != 0.9:
        vllm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
    if cpu_offload_gb > 0:
        vllm_kwargs["cpu_offload_gb"] = cpu_offload_gb
    if enforce_eager:
        vllm_kwargs["enforce_eager"] = True
    model = BaseQwenInference(
        model_name=model_name,
        device=device,
        torch_dtype=torch_dtype,
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
        tensor_parallel_size=tensor_parallel_size,
        **vllm_kwargs,
    )

    pbar = tqdm(total=total_inferences, desc="Extracting TOM values")
    metadata = {
        "source_input": str(input_path),
        "tom_extracted_at": datetime.now().isoformat(),
        "model": model_name,
        "analysis_type": "TOM (Theory of Mind)",
        "analysis_mode": "two_run",  # part1: beliefs+desires+intentions, part2: emotions+knowledge+perspective_taking
    }
    if random_sample is not None:
        metadata["random_sample"] = random_sample
        metadata["random_sample_of_total"] = total_available
        if seed is not None:
            metadata["random_seed"] = seed

    for idx, sample in enumerate(samples):
        post = sample.get("post", {})
        post_id = _post_key(post)
        post_title = post.get("title", "")
        post_body = post.get("selftext", "")
        comment_targets = _get_comment_targets(sample)

        # 1. Post TOM (skip if already in output) — 2 runs: part1 + part2
        if post_id not in out_posts:
            post_for_prompt = {"title": post_title, "selftext": post_body}
            try:
                # Part 1: analysis, beliefs, desires, intentions
                prompt1 = build_tom_post_prompt(post_for_prompt, TOM_POST_PROMPT_PART1)
                output1 = model.generate(prompt1)
                text1 = output1.get("raw_text") or output1.get("response") or ""
                parsed1 = parse_tom_response(text1)
                pbar.update(1)
                # Part 2: emotions, knowledge, perspective_taking
                prompt2 = build_tom_post_prompt(post_for_prompt, TOM_POST_PROMPT_PART2)
                output2 = model.generate(prompt2)
                text2 = output2.get("raw_text") or output2.get("response") or ""
                parsed2 = parse_tom_response(text2)
                pbar.update(1)
                merged = merge_parsed_tom(parsed1, parsed2)
                tom = {
                    "analysis": merged.get("analysis"),
                    "beliefs": merged.get("beliefs"),
                    "desires": merged.get("desires"),
                    "intentions": merged.get("intentions"),
                    "emotions": merged.get("emotions"),
                    "knowledge": merged.get("knowledge"),
                    "perspective_taking": merged.get("perspective_taking"),
                }
                if merged.get("truncated"):
                    tom["truncated"] = True
                    tom["truncated_categories"] = merged.get("truncated_categories", [])
                out_posts[post_id] = {"id": post_id, "tom": tom, "raw_response_1": text1, "raw_response_2": text2}
            except Exception as e:
                out_posts[post_id] = {"id": post_id, "tom": None, "tom_error": str(e)}
                pbar.update(2)  # skip both parts
            if post_id.startswith("p_"):
                out_posts[post_id]["text_preview"] = (post_title or "")[:120] + ("..." if len(post_title or "") > 120 else "")
        else:
            pbar.update(2)

        # 2. Comment TOM (skip if already in output) — 2 runs per comment: part1 + part2
        for comment_idx, (target, comment) in enumerate(comment_targets):
            c_key = _comment_key(comment, post_id)
            if c_key not in out_comments:
                try:
                    # Part 1: analysis, beliefs, desires, intentions
                    prompt1 = build_tom_comment_prompt(post_title, comment, TOM_COMMENT_PROMPT_PART1)
                    output1 = model.generate(prompt1)
                    text1 = output1.get("raw_text") or output1.get("response") or ""
                    parsed1 = parse_tom_response(text1)
                    pbar.update(1)
                    # Part 2: emotions, knowledge, perspective_taking
                    prompt2 = build_tom_comment_prompt(post_title, comment, TOM_COMMENT_PROMPT_PART2)
                    output2 = model.generate(prompt2)
                    text2 = output2.get("raw_text") or output2.get("response") or ""
                    parsed2 = parse_tom_response(text2)
                    pbar.update(1)
                    merged = merge_parsed_tom(parsed1, parsed2)
                    tom = {
                        "analysis": merged.get("analysis"),
                        "beliefs": merged.get("beliefs"),
                        "desires": merged.get("desires"),
                        "intentions": merged.get("intentions"),
                        "emotions": merged.get("emotions"),
                        "knowledge": merged.get("knowledge"),
                        "perspective_taking": merged.get("perspective_taking"),
                    }
                    if merged.get("truncated"):
                        tom["truncated"] = True
                        tom["truncated_categories"] = merged.get("truncated_categories", [])
                    out_comments[c_key] = {"id": c_key, "post_id": post_id, "tom": tom, "raw_response_1": text1, "raw_response_2": text2}
                except Exception as e:
                    out_comments[c_key] = {"id": c_key, "post_id": post_id, "tom": None, "tom_error": str(e)}
                    pbar.update(2)
                if c_key.startswith("c_"):
                    out_comments[c_key]["text_preview"] = (comment.get("body") or "")[:120] + ("..." if len(comment.get("body") or "") > 120 else "")
            else:
                pbar.update(2)

        # Save after each sample (incremental; output file only)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"metadata": metadata, "posts": out_posts, "comments": out_comments}, f, indent=2, ensure_ascii=False)

    pbar.close()

    # Summary
    print(f"\n{'=' * 60}")
    print(f"TOM results saved to {output_path}")
    print(f"Summary: {len(out_posts)} posts, {len(out_comments)} comments")
    print(f"{'=' * 60}")

    return {"metadata": metadata, "posts": out_posts, "comments": out_comments}


def main():
    parser = argparse.ArgumentParser(description="Extract values for delta samples")
    parser.add_argument(
        "--input", "-i",
        default="data/cmv_analysis_subset.json",
        help="Input CMV JSON (posts with persuasive / hard / easy comments; run from repo root)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="FILE",
        help="Output file for TOM results (default: input stem + _tom_values, e.g. data.json -> data_tom_values.json)"
    )
    parser.add_argument(
        "--model", "-m",
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Model name"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of samples to process (first N)"
    )
    parser.add_argument(
        "--random-sample", "-r",
        type=int,
        default=None,
        metavar="N",
        help="Randomly sample N samples (e.g. -r 4000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for --random-sample (reproducible sampling)"
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=8000,
        help="Maximum tokens to generate per response (default: 8000 for TOM analysis)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type (default: auto)"
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=True,
        help="Enable thinking mode (default: True)"
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable thinking mode"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        metavar="N",
        help="Number of GPUs for tensor parallelism (default: 1)."
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=12000,
        metavar="N",
        help="Maximum context length (default: 12000). Lower to save GPU memory."
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory for model+KV cache (default: 0.9). Raise to 0.95 to squeeze more."
    )
    parser.add_argument(
        "--cpu-offload-gb",
        type=float,
        default=0,
        help="GiB of model weights to offload to CPU per GPU (default: 0). Use when weights don't fit in GPU memory."
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph capture. Use if run hangs during model init (after 'nccl'); slightly slower but more stable."
    )
    args = parser.parse_args()

    enable_thinking = args.enable_thinking and not args.no_thinking

    print("=" * 60)
    print("Delta Samples TOM (Theory of Mind) Extraction")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Model: {args.model}")
    print(f"Tensor parallel size: {args.tensor_parallel_size} GPU(s)")
    print(f"Max model len: {args.max_model_len}")
    if args.max_model_len < 4096:
        print(f"WARNING: max_model_len={args.max_model_len} is very low. TOM prompts are ~1500+ tokens; output will be truncated. Use 8192+ to avoid truncation.")
    if args.max_model_len >= 8192 and not args.enforce_eager:
        print("If run hangs after 'nccl', stop and retry with --enforce-eager")
    if args.cpu_offload_gb > 0:
        print(f"CPU offload: {args.cpu_offload_gb} GiB")
    if args.enforce_eager:
        print("Enforce eager: True (CUDA graphs disabled)")
    if args.random_sample:
        print(f"Random sample: {args.random_sample} samples" + (f" (seed={args.seed})" if args.seed else ""))
    elif args.limit:
        print(f"Limit: {args.limit} samples")
    if args.output:
        print(f"Output: {args.output}")
    else:
        inp = Path(args.input)
        print(f"Output: {inp.parent / (inp.stem + '_tom_values' + inp.suffix)} (default)")
    print("=" * 60)

    run_inference_on_delta_samples(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        device=args.device,
        torch_dtype=args.dtype,
        max_new_tokens=args.max_tokens,
        limit=args.limit,
        random_sample=args.random_sample,
        seed=args.seed,
        enable_thinking=enable_thinking,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cpu_offload_gb=args.cpu_offload_gb,
        enforce_eager=args.enforce_eager,
    )


if __name__ == "__main__":
    main()
