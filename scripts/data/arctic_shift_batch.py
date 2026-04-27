#!/usr/bin/env python3
"""
Arctic Shift Batch Downloader
Download data from multiple subreddits in batch
"""

import json
import time
from pathlib import Path
from typing import List, Dict
from arctic_shift_downloader import ArcticShiftDownloader


def batch_download(
    subreddits: List[str],
    data_type: str = "both",
    start_date: str = None,
    end_date: str = None,
    output_dir: str = "reddit_data_batch",
    delay: float = 1.0
):
    """
    Download data from multiple subreddits
    
    Args:
        subreddits: List of subreddit names
        data_type: "posts", "comments", or "both"
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory
        delay: Delay between subreddits in seconds
    """
    downloader = ArcticShiftDownloader(output_dir=output_dir)
    results = {}
    
    for i, subreddit in enumerate(subreddits, 1):
        print(f"\n[{i}/{len(subreddits)}] Processing r/{subreddit}...")
        results[subreddit] = {}
        
        try:
            if data_type in ["posts", "both"]:
                posts = downloader.download_posts(
                    subreddit=subreddit,
                    start_date=start_date,
                    end_date=end_date
                )
                results[subreddit]["posts"] = len(posts)
            
            if data_type in ["comments", "both"]:
                comments = downloader.download_comments(
                    subreddit=subreddit,
                    start_date=start_date,
                    end_date=end_date
                )
                results[subreddit]["comments"] = len(comments)
                
        except Exception as e:
            print(f"Error processing r/{subreddit}: {e}")
            results[subreddit]["error"] = str(e)
        
        # Wait between subreddits
        if i < len(subreddits):
            print(f"Waiting {delay}s before next subreddit...")
            time.sleep(delay)
    
    # Save summary
    summary_path = Path(output_dir) / "download_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Batch download complete! Summary saved to {summary_path}")
    return results


def load_subreddit_list(filepath: str) -> List[str]:
    """Load subreddit list from a text file (one per line)"""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch download from multiple subreddits")
    parser.add_argument(
        "-f", "--file",
        help="File containing subreddit names (one per line)"
    )
    parser.add_argument(
        "-s", "--subreddits",
        nargs="+",
        help="Space-separated list of subreddits"
    )
    parser.add_argument(
        "-t", "--type",
        choices=["posts", "comments", "both"],
        default="both"
    )
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("-o", "--output", default="reddit_data_batch")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between subreddits (seconds)")
    
    args = parser.parse_args()
    
    # Get subreddit list
    if args.file:
        subreddits = load_subreddit_list(args.file)
    elif args.subreddits:
        subreddits = args.subreddits
    else:
        parser.error("Must provide either --file or --subreddits")
    
    print(f"Downloading from {len(subreddits)} subreddits...")
    
    batch_download(
        subreddits=subreddits,
        data_type=args.type,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output,
        delay=args.delay
    )
