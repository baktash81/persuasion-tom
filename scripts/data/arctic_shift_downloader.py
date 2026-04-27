#!/usr/bin/env python3
"""
Arctic Shift Reddit Data Downloader
Downloads posts and comments from specified subreddits directly to your server
"""

import requests
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional


class ArcticShiftDownloader:
    """Download Reddit data from Arctic Shift API"""
    
    BASE_URL = "https://arctic-shift.photon-reddit.com/api"
    
    def __init__(self, output_dir: str = "reddit_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        
    def download_posts(
        self, 
        subreddit: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
        output_file: Optional[str] = None
    ):
        """
        Download posts from a subreddit
        
        Args:
            subreddit: Name of the subreddit (without r/)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Number of posts to retrieve per request (max 100)
            output_file: Custom output filename
        """
        print(f"Downloading posts from r/{subreddit}...")
        
        params = {
            "subreddit": subreddit,
            "limit": min(limit, 100),  # API max is 100
        }
        
        if start_date:
            params["after"] = start_date
        if end_date:
            params["before"] = end_date
            
        all_posts = []
        search_after = None
        
        while True:
            if search_after:
                params["search_after"] = search_after
                
            try:
                response = self.session.get(
                    f"{self.BASE_URL}/posts/search",
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                
                # Check rate limiting
                if "X-RateLimit-Remaining" in response.headers:
                    remaining = response.headers["X-RateLimit-Remaining"]
                    print(f"Rate limit remaining: {remaining}")
                
                data = response.json()
                posts = data.get("data", [])
                
                if not posts:
                    break
                    
                all_posts.extend(posts)
                print(f"Downloaded {len(all_posts)} posts so far...")
                
                # Check if there are more results
                if "search_after" in data:
                    search_after = data["search_after"]
                    time.sleep(0.5)  # Be nice to the API
                else:
                    break
                    
            except requests.exceptions.RequestException as e:
                print(f"Error downloading posts: {e}")
                break
        
        # Save to file
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"posts_{subreddit}_{timestamp}.json"
            
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_posts, f, indent=2, ensure_ascii=False)
            
        print(f"✓ Downloaded {len(all_posts)} posts to {output_path}")
        return all_posts
    
    def download_comments(
        self,
        subreddit: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
        output_file: Optional[str] = None
    ):
        """
        Download comments from a subreddit
        
        Args:
            subreddit: Name of the subreddit (without r/)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Number of comments to retrieve per request (max 100)
            output_file: Custom output filename
        """
        print(f"Downloading comments from r/{subreddit}...")
        
        params = {
            "subreddit": subreddit,
            "limit": min(limit, 100),  # API max is 100
        }
        
        if start_date:
            params["after"] = start_date
        if end_date:
            params["before"] = end_date
            
        all_comments = []
        search_after = None
        
        while True:
            if search_after:
                params["search_after"] = search_after
                
            try:
                response = self.session.get(
                    f"{self.BASE_URL}/comments/search",
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                
                # Check rate limiting
                if "X-RateLimit-Remaining" in response.headers:
                    remaining = response.headers["X-RateLimit-Remaining"]
                    print(f"Rate limit remaining: {remaining}")
                
                data = response.json()
                comments = data.get("data", [])
                
                if not comments:
                    break
                    
                all_comments.extend(comments)
                print(f"Downloaded {len(all_comments)} comments so far...")
                
                # Check if there are more results
                if "search_after" in data:
                    search_after = data["search_after"]
                    time.sleep(0.5)  # Be nice to the API
                else:
                    break
                    
            except requests.exceptions.RequestException as e:
                print(f"Error downloading comments: {e}")
                break
        
        # Save to file
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"comments_{subreddit}_{timestamp}.json"
            
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_comments, f, indent=2, ensure_ascii=False)
            
        print(f"✓ Downloaded {len(all_comments)} comments to {output_path}")
        return all_comments


def main():
    parser = argparse.ArgumentParser(
        description="Download Reddit data from Arctic Shift API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download posts from r/python
  python arctic_shift_downloader.py -s python -t posts
  
  # Download comments from r/machinelearning with date range
  python arctic_shift_downloader.py -s machinelearning -t comments --start 2024-01-01 --end 2024-12-31
  
  # Download both posts and comments
  python arctic_shift_downloader.py -s datascience -t both
  
  # Specify custom output directory
  python arctic_shift_downloader.py -s AI -t posts -o /path/to/output
        """
    )
    
    parser.add_argument(
        "-s", "--subreddit",
        required=True,
        help="Subreddit name (without r/)"
    )
    
    parser.add_argument(
        "-t", "--type",
        choices=["posts", "comments", "both"],
        default="both",
        help="Type of data to download (default: both)"
    )
    
    parser.add_argument(
        "--start",
        help="Start date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end",
        help="End date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="reddit_data",
        help="Output directory (default: reddit_data)"
    )
    
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=100,
        help="Items per request (max 100, default: 100)"
    )
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = ArcticShiftDownloader(output_dir=args.output)
    
    # Download data
    if args.type in ["posts", "both"]:
        downloader.download_posts(
            subreddit=args.subreddit,
            start_date=args.start,
            end_date=args.end,
            limit=args.limit
        )
    
    if args.type in ["comments", "both"]:
        downloader.download_comments(
            subreddit=args.subreddit,
            start_date=args.start,
            end_date=args.end,
            limit=args.limit
        )
    
    print("\n✓ Download complete!")


if __name__ == "__main__":
    main()
