#!/usr/bin/env python3
"""
Clean posts_with_deltas.json: filter out deleted posts/comments, add full post info
and 10 randomly selected non-deleted comments per post.

Output: JSON with posts that have valid (non-deleted) post, persuasive_comment,
and 10 random non-deleted comments.
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from typing import Optional

# Reddit deleted/removed indicators
DELETED_AUTHORS = {'[deleted]', '[removed]', '', None}
DELETED_BODIES = {'[deleted]', '[removed]', '[deleted by user]', '[removed by Reddit]', '', None}


def is_deleted_author(author) -> bool:
    """Check if author field indicates deleted content."""
    if author is None:
        return True
    s = str(author).strip().lower()
    return s in ('[deleted]', '[removed]', '')


def is_deleted_body(body) -> bool:
    """Check if body/text field indicates deleted content."""
    if body is None:
        return True
    s = str(body).strip().lower()
    return s in ('[deleted]', '[removed]', '[deleted by user]', '[removed by reddit]', '')


def is_post_deleted(post: dict) -> bool:
    """Check if post is deleted (author or title/selftext removed)."""
    if is_deleted_author(post.get('author')):
        return True
    title = post.get('title') or ''
    selftext = post.get('selftext') or ''
    if is_deleted_body(title) or (title.strip().lower() in DELETED_BODIES):
        return True
    if is_deleted_body(selftext) or (selftext.strip().lower() in DELETED_BODIES):
        return True
    return False


def is_comment_deleted(comment: dict) -> bool:
    """Check if comment is deleted (author or body removed)."""
    if is_deleted_author(comment.get('author')):
        return True
    if is_deleted_body(comment.get('body')):
        return True
    return False


def load_posts_index(posts_path: Path) -> dict:
    """Load posts: link_id -> full post dict."""
    posts = {}
    path_str = str(posts_path)

    if path_str.endswith('.jsonl'):
        with open(posts_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading posts", unit=" posts"):
                post = json.loads(line)
                link_id = post.get('link_id') or f"t3_{post.get('id', '')}"
                if not link_id.startswith('t3_'):
                    link_id = f"t3_{post.get('id', '')}"
                posts[link_id] = post
    else:
        with open(posts_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        items = data if isinstance(data, list) else data.get('posts', data.get('data', [data]))
        for post in tqdm(items, desc="Loading posts", unit=" posts"):
            link_id = post.get('link_id') or f"t3_{post.get('id', '')}"
            if not link_id.startswith('t3_'):
                link_id = f"t3_{post.get('id', '')}"
            posts[link_id] = post

    print(f"  Loaded {len(posts):,} posts")
    return posts


def load_comments_by_post(comments_path: Path, post_link_ids: set, total_comments: Optional[int] = None) -> dict:
    """Load all comments for given posts: link_id -> list of comments."""
    post_comments = defaultdict(list)
    with open(comments_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_comments, desc="Loading comments", unit=" comments"):
            try:
                comment = json.loads(line)
            except json.JSONDecodeError:
                continue
            link_id = comment.get('link_id', '')
            if link_id in post_link_ids:
                post_comments[link_id].append(comment)
    return dict(post_comments)


def main():
    parser = argparse.ArgumentParser(description='Clean posts_with_deltas.json')
    parser.add_argument(
        '--input', '-i',
        default='results/posts_with_deltas.json',
        help='Input posts_with_deltas.json'
    )
    parser.add_argument(
        '--posts', '-p',
        default='data/reddit_data/posts_changemyview_full.jsonl',
        help='Source posts file'
    )
    parser.add_argument(
        '--comments', '-c',
        default='data/reddit_data/comments_changemyview_full.jsonl',
        help='Source comments file'
    )
    parser.add_argument(
        '--output', '-o',
        default='results/posts_with_deltas_clean.json',
        help='Output cleaned JSON'
    )
    parser.add_argument(
        '--num-comments',
        type=int,
        default=10,
        help='Number of random comments per post (default: 10)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--total-comments',
        type=int,
        default=None,
        help='Total comments for progress bar'
    )
    args = parser.parse_args()

    random.seed(args.seed)

    input_path = Path(args.input)
    posts_path = Path(args.posts)
    comments_path = Path(args.comments)

    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        return 1

    print("=" * 60)
    print("Clean Delta Posts")
    print("=" * 60)

    # Load delta data
    with open(input_path, 'r', encoding='utf-8') as f:
        delta_data = json.load(f)

    delta_details = delta_data.get('delta_details', [])
    metadata = delta_data.get('metadata', {})

    # Get source paths from metadata if files not found
    if not posts_path.exists():
        posts_path = Path(metadata.get('source_posts', args.posts))
    if not comments_path.exists():
        comments_path = Path(metadata.get('source_comments', args.comments))

    if not posts_path.exists() or not comments_path.exists():
        print(f"Error: Source files not found. Posts: {posts_path}, Comments: {comments_path}")
        return 1

    # Group delta_details by post (keep first valid persuasive comment per post)
    post_to_deltas = defaultdict(list)
    for d in delta_details:
        post_to_deltas[d['post_link_id']].append(d)

    # Load full post data
    all_link_ids = set(post_to_deltas.keys())
    posts_index = load_posts_index(posts_path)

    # Load comments for our posts
    post_comments = load_comments_by_post(comments_path, all_link_ids, args.total_comments)

    # Build clean output
    results = []
    skipped_post_deleted = 0
    skipped_persuasive_deleted = 0
    skipped_no_post = 0
    skipped_no_comments = 0

    for link_id in tqdm(all_link_ids, desc="Building clean output"):
        post = posts_index.get(link_id)
        if not post:
            skipped_no_post += 1
            continue

        if is_post_deleted(post):
            skipped_post_deleted += 1
            continue

        deltas = post_to_deltas[link_id]
        persuasive_comment = None
        for d in deltas:
            pc = d.get('persuasive_comment', {})
            if not is_comment_deleted(pc):
                persuasive_comment = pc
                break

        if not persuasive_comment:
            skipped_persuasive_deleted += 1
            continue

        comments = post_comments.get(link_id, [])
        persuasive_id = persuasive_comment.get('id')
        delta_reply_ids = {d.get('delta_reply', {}).get('id') for d in deltas}

        # Determine if persuasive_comment is level 0 (top-level: parent_id == link_id)
        full_persuasive = next((c for c in comments if c.get('id') == persuasive_id), None)
        persuasive_is_level_zero = (
            full_persuasive.get('parent_id') == link_id if full_persuasive else None
        )
        persuasive_comment_with_flag = dict(persuasive_comment)
        persuasive_comment_with_flag['is_level_zero'] = persuasive_is_level_zero

        # Non-deleted LEVEL 0 comments only, excluding persuasive and delta replies
        valid_comments = [
            c for c in comments
            if not is_comment_deleted(c)
            and c.get('parent_id') == link_id  # level 0 only
            and c.get('id') != persuasive_id
            and c.get('id') not in delta_reply_ids
        ]

        if len(valid_comments) < args.num_comments:
            selected = valid_comments
        else:
            selected = random.sample(valid_comments, args.num_comments)

        # Include delta_reply for context (first one)
        delta_reply = deltas[0].get('delta_reply', {}) if deltas else {}

        results.append({
            'post': {
                'id': post.get('id'),
                'link_id': link_id,
                'title': post.get('title'),
                'author': post.get('author'),
                'selftext': post.get('selftext'),
                'score': post.get('score'),
                'num_comments': post.get('num_comments'),
                'created_utc': post.get('created_utc'),
                'permalink': post.get('permalink'),
                'link_flair_text': post.get('link_flair_text'),
            },
            'persuasive_comment': persuasive_comment_with_flag,
            'delta_reply': delta_reply,
            'comments': [
                {
                    'id': c.get('id'),
                    'author': c.get('author'),
                    'body': c.get('body'),
                    'score': c.get('score'),
                    'parent_id': c.get('parent_id'),
                    'is_level_zero': True,
                }
                for c in selected
            ],
        })

    output = {
        'metadata': {
            'source': str(input_path),
            'total_input_posts_with_deltas': len(all_link_ids),
            'total_clean_posts': len(results),
            'skipped_post_deleted': skipped_post_deleted,
            'skipped_persuasive_deleted': skipped_persuasive_deleted,
            'skipped_no_post_in_source': skipped_no_post,
            'num_comments_per_post': args.num_comments,
            'comments_are_level_zero': True,
            'persuasive_comment_has_is_level_zero_flag': True,
        },
        'posts': results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Clean posts output:     {len(results):,}")
    print(f"  Skipped (post deleted): {skipped_post_deleted:,}")
    print(f"  Skipped (persuasive deleted): {skipped_persuasive_deleted:,}")
    print(f"  Skipped (post not in source): {skipped_no_post:,}")
    print(f"  Output: {output_path}")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    exit(main())
