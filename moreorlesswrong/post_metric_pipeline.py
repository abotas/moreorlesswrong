import json
from pathlib import Path
from typing import List, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

from models import Post
from post_metric_registry import compute_metrics_for_post


def process_single_post(
    post: Post,
    metrics: List[str],
    version_id: str,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"]
):
    """Process a single post - compute metrics without extracting claims."""
    # Create directories
    metrics_dir = Path(f"data/post_metrics/{version_id}")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Thread] Processing post: {post.title[:50]}... (ID: {post.post_id})")
    
    # Check if metrics already exist
    metrics_file = metrics_dir / f"{post.post_id}.json"
    if metrics_file.exists():
        print(f"  Loading existing metrics from {metrics_file}")
        with open(metrics_file, 'r') as f:
            existing_metrics = json.load(f)
    else:
        existing_metrics = {}
    
    # Check which metrics are missing
    missing_metrics = []
    for metric_name in metrics:
        if metric_name not in existing_metrics:
            missing_metrics.append(metric_name)
    
    if missing_metrics:
        print(f"  Computing {len(missing_metrics)} metrics for post...")
        computed = compute_metrics_for_post(missing_metrics, post, model)
        
        # Store computed metrics, skip failed ones
        for result in computed:
            if isinstance(result, tuple):
                # This is an error tuple (metric_name, error_string)
                # Already printed in compute_metrics_for_post
                continue
            metric_name = result.metric_name()
            existing_metrics[metric_name] = result.model_dump()
        
        # Save all metrics for this post
        with open(metrics_file, 'w') as f:
            json.dump(existing_metrics, f, indent=2)
        print(f"  Saved metrics to {metrics_file}")
    else:
        print(f"  All metrics already computed")
    
    return post.post_id


def process_posts(
    posts: List[Post],
    metrics: List[str],
    version_id: str,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini",
    max_workers: int = 4
):
    """Process posts to compute metrics without extracting claims.
    
    This function is interruptable and will skip already processed data.
    
    Args:
        posts: List of posts to process
        metrics: List of metric names (strings) to compute
        version_id: Version identifier for this processing run
        model: LLM model to use
        max_workers: Maximum number of concurrent threads (default: 4)
    """
    print(f"\n{'='*60}")
    print(f"Starting post-level pipeline with {max_workers} worker threads")
    print(f"Version: {version_id}")
    print(f"Model: {model}")
    print(f"Posts to process: {len(posts)}")
    print(f"Metrics ({len(metrics)}): {', '.join(metrics)}")
    print(f"{'='*60}")
    
    successful = []
    failed = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                process_single_post, 
                post, 
                metrics, 
                version_id,
                model
            ): post 
            for post in posts
        }
        
        # Process as they complete
        for future in as_completed(futures):
            post = futures[future]
            try:
                post_id = future.result()
                successful.append(post_id)
                print(f"✓ Completed: {post.title[:30]}...")
            except Exception as e:
                failed.append((post.post_id, str(e)))
                print(f"✗ Failed: {post.title[:30]}... - Error: {e}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Pipeline complete!")
    print(f"Successful: {len(successful)}/{len(posts)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for post_id, error in failed:
            print(f"  - {post_id}: {error}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    from db import get_representative_posts
    
    parser = argparse.ArgumentParser(description="Process posts to compute metrics")
    parser.add_argument("--posts", type=int, default=10,
                       help="Number of posts to process (default: 10)")
    parser.add_argument("--version", type=str, default="post_v1",
                       help="Version ID for this processing run (default: post_v1)")
    parser.add_argument("--threads", type=int, default=4,
                       help="Number of worker threads (default: 4)")
    parser.add_argument("--model", type=str, default="gpt-5-mini",
                       choices=["gpt-5-nano", "gpt-5-mini", "gpt-5"],
                       help="Model to use (default: gpt-5-mini)")
    
    args = parser.parse_args()
    
    # Hardcoded list of metrics to compute
    METRICS = [
        "PostValue",
        "PostRobustness", 
        "PostAuthorAura",
        "PostClarity",
        "PostNovelty",
        "PostInferentialSupport",
        "PostExternalValidation"
    ]
    
    # Get posts
    posts = get_representative_posts(args.posts)
    
    # Process posts
    process_posts(
        posts=posts,
        metrics=METRICS,
        version_id=args.version,
        model=args.model,
        max_workers=args.threads
    )