"""
CMV Data Inference Script

Runs LLaMA and Qwen models on CMV posts with their top comments.
Backend: vLLM (offline inference).
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

from vllm import LLM, SamplingParams
from tqdm import tqdm


# ============================================================================
# PROMPT TEMPLATE - CUSTOMIZE THIS
# ============================================================================

PROMPT_TEMPLATE = """
Your task is to analyze the user's (human's) messages in this conversation and 
identify which values they actively express or demonstrate.
<conversation>
Post Title: 
{title}
Post Body: 
{body}
Comments: 
{comments}
</conversation>

Look for values the users directly or explicitly expresses through their statements. Focus
on what the user explicitly states about their beliefs, preferences, or intentions.

Do NOT count instances where the user merely:
+ Asks for technical help or factual information, even on values-relevant topics, without expressing a value judgment
+ Shares values-laden text for editing/review that they didn't actually write themselves

Summarize each value in 1-4 words that are as accurate and precise as possible. Only use
commas to separate the values (i.e. format the answer as ''x, y, z'' where x, y, z
are different values). If no values are clearly demonstrated, write ''none''.

Output Format:

<thinking>
2-3 short and concise sentences thinking through the values
</thinking>

<answer>
Selected value(s), comma-separated without quote marks, or ''none''
</answer>

Do NOT put any explanation within the <answer> tags, only the final values.
"""

# ============================================================================


def load_data(filepath: str) -> list:
    """Load CMV data from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("posts", [])


def format_comments(comments: list) -> str:
    """Format comments for the prompt."""
    formatted = []
    for i, comment in enumerate(comments[:5], 1):
        score = comment.get("score", 0)
        body = comment.get("body", "")
        formatted.append(f"Comment {i} (score: {score}):\n{body}")
    return "\n\n".join(formatted)


def build_prompt(post: dict, template: str = PROMPT_TEMPLATE) -> str:
    """Build the prompt for a single post."""
    title = post.get("title", "")
    body = post.get("selftext", "")
    comments = format_comments(post.get("comments", []))
    
    return template.format(
        title=title,
        body=body,
        comments=comments,
    )


# ============================================================================
# LLAMA INFERENCE (vLLM)
# ============================================================================

def _vllm_dtype(torch_dtype: str):
    """Map torch_dtype string to vLLM dtype (vLLM uses 'auto', 'float16', etc.)."""
    if torch_dtype in ("auto", "float16", "bfloat16", "float32"):
        return torch_dtype
    return "auto"


class LlamaInference:
    """Inference wrapper for LLaMA models using vLLM."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        **vllm_kwargs,
    ):
        """
        Initialize LLaMA model with vLLM.

        Args:
            model_name: HuggingFace model name or path
            device: Ignored (vLLM uses CUDA by default); kept for API compatibility
            torch_dtype: Data type ("auto", "float16", "bfloat16", "float32")
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling (vLLM always samples when temperature > 0)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            trust_remote_code: Allow custom model code
            **vllm_kwargs: Additional arguments passed to vLLM LLM()
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

        print(f"Loading LLaMA model with vLLM: {model_name}")
        self.llm = LLM(
            model=model_name,
            dtype=_vllm_dtype(torch_dtype),
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            **vllm_kwargs,
        )
        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p,
        )
        print("LLaMA model loaded (vLLM).")

    def generate(self, prompt: str) -> dict:
        """
        Generate response for a prompt.

        Returns:
            dict with 'response' key (and optionally 'thinking' for compatibility)
        """
        outputs = self.llm.generate([prompt], self.sampling_params)
        text = outputs[0].outputs[0].text
        stripped = text.strip()
        return {
            "response": stripped,
            "thinking": None,
            "raw_text": stripped,
        }


# ============================================================================
# QWEN INFERENCE (vLLM)
# ============================================================================

def _parse_qwen_thinking(raw_text: str, enable_thinking: bool) -> tuple[Optional[str], str]:
    """
    Split vLLM Qwen output into thinking and content using </think>.
    When enable_thinking is False, treat all output as content.
    """
    if not enable_thinking:
        return None, (raw_text or "").strip()
    if "</think>" not in raw_text:
        # Truncated or no thinking: treat everything as thinking or content
        return (raw_text or "").strip(), ""
    before, _, after = raw_text.partition("</think>")
    thinking = before.strip() if before.strip() else None
    content = after.strip() if after.strip() else ""
    return thinking, content


class QwenInference:
    """Inference wrapper for Qwen3 models with thinking mode support (vLLM)."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 1024,
        enable_thinking: bool = True,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        **vllm_kwargs,
    ):
        """
        Initialize Qwen3 model with vLLM.

        Args:
            model_name: HuggingFace model name or path (e.g., "Qwen/Qwen3-8B")
            device: Ignored (vLLM uses CUDA); kept for API compatibility
            torch_dtype: Data type ("auto", "float16", "bfloat16", "float32")
            max_new_tokens: Maximum tokens to generate
            enable_thinking: Enable thinking mode (default: True)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            trust_remote_code: Allow custom model code
            **vllm_kwargs: Additional arguments passed to vLLM LLM()
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking

        print(f"Loading Qwen model with vLLM: {model_name}")
        self.llm = LLM(
            model=model_name,
            dtype=_vllm_dtype(torch_dtype),
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            **vllm_kwargs,
        )
        self.sampling_params = SamplingParams(max_tokens=max_new_tokens)
        self._chat_kwargs = {"enable_thinking": enable_thinking}
        print("Qwen model loaded (vLLM).")

    def generate(self, prompt: str) -> dict:
        """
        Generate response for a prompt using vLLM chat API.

        Returns:
            dict with 'response' and 'thinking' keys
        """
        messages = [{"role": "user", "content": prompt}]
        outputs = self.llm.chat(
            [messages],
            self.sampling_params,
            chat_template_kwargs=self._chat_kwargs,
            use_tqdm=False,
        )
        raw_text = outputs[0].outputs[0].text
        thinking, content = _parse_qwen_thinking(raw_text, self.enable_thinking)
        return {
            "response": content,
            "thinking": thinking,
            "raw_text": raw_text,  # full model output before any split (for debugging/logging)
        }


# ============================================================================
# INFERENCE RUNNER
# ============================================================================

def run_inference(
    posts: list,
    model,  # LlamaInference or QwenInference
    output_path: str,
    prompt_template: str = PROMPT_TEMPLATE,
    limit: Optional[int] = None,
    start_idx: int = 0,
    save_thinking: bool = False,
):
    """
    Run inference on posts.
    
    Args:
        posts: List of posts
        model: Model inference wrapper (LlamaInference or QwenInference)
        output_path: Path to save results
        prompt_template: Prompt template to use
        limit: Maximum number of posts to process
        start_idx: Starting index (for resuming)
        save_thinking: Whether to save thinking content (for Qwen)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing results if any
    results = []
    processed_ids = set()
    
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
            results = existing.get("results", [])
            processed_ids = {r["post_id"] for r in results}
            print(f"Loaded {len(results)} existing results")
    
    # Filter posts
    posts_to_process = posts[start_idx:]
    if limit:
        posts_to_process = posts_to_process[:limit]
    
    # Skip already processed
    posts_to_process = [p for p in posts_to_process if p["id"] not in processed_ids]
    
    print(f"Processing {len(posts_to_process)} posts...")
    
    for post in tqdm(posts_to_process, desc="Inference"):
        try:
            prompt = build_prompt(post, prompt_template)
            
            start_time = time.time()
            output = model.generate(prompt)
            inference_time = time.time() - start_time
            
            result = {
                "post_id": post["id"],
                "title": post["title"],
                "model": model.model_name,
                "response": output["response"],
                "inference_time": inference_time,
                "timestamp": datetime.now().isoformat(),
            }
            
            # Optionally save thinking content
            if save_thinking and output.get("thinking"):
                result["thinking"] = output["thinking"]
            
            results.append(result)
            
            # Save incrementally
            save_results(results, output_path, model.model_name)
            
        except Exception as e:
            print(f"Error processing post {post['id']}: {e}")
            continue
    
    return results


def save_results(results: list, output_path: Path, model_name: str):
    """Save results to JSON file."""
    data = {
        "metadata": {
            "model": model_name,
            "total_results": len(results),
            "last_updated": datetime.now().isoformat(),
        },
        "results": results,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Run inference on CMV data")
    
    # Data arguments
    parser.add_argument(
        "--input", "-i", type=str, default="data/cmv_data.json",
        help="Input CMV data file (default: data/cmv_data.json)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file for results (default: data/inference_{model}.json)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model", "-m", type=str, required=True,
        help="Model name or path (e.g., 'meta-llama/Llama-3-8B', 'Qwen/Qwen3-8B')"
    )
    parser.add_argument(
        "--model-type", "-t", type=str, choices=["llama", "qwen"], default=None,
        help="Model type: 'llama' or 'qwen' (auto-detected if not specified)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--dtype", type=str, default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type (default: auto)"
    )
    
    # Generation arguments
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Maximum new tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature for LLaMA (default: 0.7)"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9,
        help="Nucleus sampling p for LLaMA (default: 0.9)"
    )
    
    # Qwen-specific arguments
    parser.add_argument(
        "--enable-thinking", action="store_true", default=True,
        help="Enable thinking mode for Qwen (default: True)"
    )
    parser.add_argument(
        "--no-thinking", action="store_true",
        help="Disable thinking mode for Qwen"
    )
    parser.add_argument(
        "--save-thinking", action="store_true",
        help="Save thinking content in results (for Qwen)"
    )
    
    # Processing arguments
    parser.add_argument(
        "--limit", "-l", type=int, default=None,
        help="Limit number of posts to process"
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Starting index (default: 0)"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, metavar="N",
        help="Number of GPUs for tensor parallelism (default: 1). Use 3 to run on 3 GPUs."
    )

    args = parser.parse_args()
    
    # Auto-detect model type
    if args.model_type is None:
        model_lower = args.model.lower()
        if "qwen" in model_lower:
            args.model_type = "qwen"
        else:
            args.model_type = "llama"
        print(f"Auto-detected model type: {args.model_type}")
    
    # Handle thinking flag
    enable_thinking = args.enable_thinking and not args.no_thinking
    
    # Set default output path
    if args.output is None:
        model_short = args.model.split("/")[-1].lower()
        args.output = f"data/inference_{model_short}.json"
    
    print("=" * 60)
    print("CMV Inference")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Model type: {args.model_type}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Tensor parallel size: {args.tensor_parallel_size} GPU(s)")
    if args.model_type == "qwen":
        print(f"Thinking mode: {enable_thinking}")
    else:
        print(f"Temperature: {args.temperature}")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from {args.input}...")
    posts = load_data(args.input)
    print(f"Loaded {len(posts)} posts")
    
    # Initialize model based on type
    print(f"\nInitializing {args.model_type.upper()} model...")
    
    if args.model_type == "qwen":
        model = QwenInference(
            model_name=args.model,
            device=args.device,
            torch_dtype=args.dtype,
            max_new_tokens=args.max_tokens,
            enable_thinking=enable_thinking,
            tensor_parallel_size=args.tensor_parallel_size,
        )
    else:  # llama
        model = LlamaInference(
            model_name=args.model,
            device=args.device,
            torch_dtype=args.dtype,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            tensor_parallel_size=args.tensor_parallel_size,
        )
    
    # Run inference
    print(f"\nStarting inference...")
    results = run_inference(
        posts=posts,
        model=model,
        output_path=args.output,
        limit=args.limit,
        start_idx=args.start,
        save_thinking=args.save_thinking,
    )
    
    print(f"\n" + "=" * 60)
    print(f"Completed! Processed {len(results)} posts")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
