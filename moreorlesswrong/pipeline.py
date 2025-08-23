import json
from pathlib import Path
from typing import List, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

from models import Post, Claim
from claim_extractor import extract_claims
from claim_metric_registry import compute_metrics_for_claim


def process_single_post(
    post: Post,
    metrics: List[str],
    version_id: str,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"],
    claims_per_post: int
):
    """Process a single post - extract claims and compute metrics."""
    # Create directories
    claims_dir = Path(f"data/claims/{version_id}")
    metrics_dir = Path(f"data/metrics/{version_id}")
    claims_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Thread] Processing post: {post.title[:50]}... (ID: {post.post_id})")
    
    # Check if claims already exist
    claims_file = claims_dir / f"{post.post_id}.json"
    if claims_file.exists():
        print(f"  Loading existing claims from {claims_file}")
        with open(claims_file, 'r') as f:
            claims_data = json.load(f)
            claims = [Claim(**c) for c in claims_data]
    else:
        # Extract claims
        print(f"  Extracting {claims_per_post} claims...")
        claims = extract_claims(post, claims_per_post, model=model)
        
        # Save claims
        with open(claims_file, 'w') as f:
            json.dump([c.model_dump() for c in claims], f, indent=2)
        print(f"  Saved {len(claims)} claims to {claims_file}")
    
    # Check if metrics already exist
    metrics_file = metrics_dir / f"{post.post_id}.json"
    if metrics_file.exists():
        print(f"  Loading existing metrics from {metrics_file}")
        with open(metrics_file, 'r') as f:
            existing_metrics = json.load(f)
    else:
        existing_metrics = {}
    
    # Compute missing metrics for each claim
    all_metrics_for_post = existing_metrics.copy()
    
    for claim in claims:
        if claim.claim_id not in all_metrics_for_post:
            all_metrics_for_post[claim.claim_id] = {}
        
        # Check which metrics are missing for this claim
        missing_metrics = []
        for metric_name in metrics:
            if metric_name not in all_metrics_for_post[claim.claim_id]:
                missing_metrics.append(metric_name)
        
        if missing_metrics:
            print(f"  Computing {len(missing_metrics)} metrics for claim {claim.claim_id}...")
            computed = compute_metrics_for_claim(missing_metrics, claim, post, model)
            
            # Store computed metrics
            for metric_name, metric_obj in computed.items():
                all_metrics_for_post[claim.claim_id][metric_name] = metric_obj.model_dump()
    
    # Save all metrics for this post
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics_for_post, f, indent=2)
    print(f"  Saved metrics to {metrics_file}")
    
    return post.post_id


def process_posts(
    posts: List[Post],
    metrics: List[str],
    version_id: str,
    claims_per_post: int,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini",
    max_workers: int = 4
):
    """Process posts to extract claims and compute metrics with multithreading.
    
    This function is interruptable and will skip already processed data.
    
    Args:
        posts: List of posts to process
        metrics: List of metric names (strings) to compute
        version_id: Version identifier for this processing run
        claims_per_post: Number of claims to extract per post
        model: LLM model to use
        max_workers: Maximum number of concurrent threads (default: 4)
    """
    print(f"Starting parallel processing with {max_workers} workers...")
    print(f"Processing {len(posts)} posts with metrics: {metrics}")
    
    completed_posts = []
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_post = {
            executor.submit(
                process_single_post,
                post,
                metrics,
                version_id,
                model,
                claims_per_post
            ): post
            for post in posts
        }
        
        # Process completed tasks
        for future in as_completed(future_to_post):
            post = future_to_post[future]
            try:
                post_id = future.result()
                completed_posts.append(post_id)
                print(f"✓ Completed processing post: {post_id}")
            except Exception as exc:
                print(f"✗ Post {post.post_id} generated an exception: {exc}")
    
    print(f"\n✅ Processing complete for {len(completed_posts)}/{len(posts)} posts")
    print(f"  Claims saved to: data/claims/{version_id}/")
    print(f"  Metrics saved to: data/metrics/{version_id}/")


if __name__ == "__main__":
    import argparse
    from db import get_representative_posts
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run metrics pipeline on representative posts')
    parser.add_argument('--claims_per_post', type=int, default=1, help='Number of claims to extract per post (default: 1)')
    parser.add_argument('--version_id', type=str, default='v1', help='Version identifier for this processing run (default: v1)')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads for parallel processing (default: 4)')
    parser.add_argument('--model', type=str, default='gpt-5-mini', 
                       choices=['gpt-5-nano', 'gpt-5-mini', 'gpt-5'],
                       help='Model to use for evaluation (default: gpt-5-mini)')
    
    args = parser.parse_args()
    
    # Get representative posts (n=10)
    print("Fetching representative posts (n=10)...")
    posts = get_representative_posts(20)
    print(f"Found {len(posts)} posts")
    
    print(f"\nStarting pipeline with {args.threads} threads using {args.model}...")
    process_posts(
        posts=posts,
        metrics=[
            "Novelty", 
            "InferentialSupport",
            "ExternalValidation", 
            "AuthorAura",
            "Robustness",
            "Value",
            "NoveltySupport",
        ],
        version_id=args.version_id,
        claims_per_post=args.claims_per_post,
        model=args.model,
        max_workers=args.threads
    )
    
    print("\n✅ Pipeline run complete!")