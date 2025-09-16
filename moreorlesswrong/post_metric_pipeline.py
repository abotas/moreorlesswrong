import json
from pathlib import Path
from typing import List, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

from models import Post
from metric_registry_v2 import compute_metrics
from metric_protocol import MetricContext

# Import metrics to trigger auto-registration
import post_metrics.v0  # This imports all V0 metrics
import post_metrics.v2  # This imports all V2 metrics
import post_metrics.v3  # This imports all V3 metrics


def process_single_post(
    post: Post,
    metrics: List[str],
    version_id: str,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"],
    n_related_posts: int = 5
):
    """Process a single post using the new class-based metric system."""
    # Create directories
    metrics_dir = Path(f"./data/post_metrics/{version_id}")
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

        # Create context
        context = MetricContext(
            model=model,
            n_related_posts=n_related_posts
        )

        try:
            # Compute all missing metrics at once
            computed_metrics = compute_metrics(post, missing_metrics, context)

            # Store computed metrics
            for metric_name, metric_result in computed_metrics.items():
                if metric_result is not None:
                    existing_metrics[metric_name] = metric_result.model_dump()
                    print(f"    ✓ {metric_name}")
                else:
                    print(f"    ✗ {metric_name} failed")

        except Exception as e:
            print(f"    ERROR computing metrics: {e}")

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
    max_workers: int = 4,
    n_related_posts: int = 5
):
    """Process posts to compute metrics using the new class-based system.

    This function is interruptable and will skip already processed data.

    Args:
        posts: List of posts to process
        metrics: List of metric names (strings) to compute
        version_id: Version identifier for this processing run
        model: LLM model to use
        max_workers: Maximum number of concurrent threads (default: 4)
        n_related_posts: Number of related posts to use for synthesis context (default: 5)
    """
    print(f"\n{'='*60}")
    print(f"Starting post-level pipeline with {max_workers} worker threads")
    print(f"Version: {version_id}")
    print(f"Model: {model}")
    print(f"Posts to process: {len(posts)}")
    print(f"Metrics ({len(metrics)}): {', '.join(metrics)}")
    print(f"Related Posts for Context: {n_related_posts}")
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
                model,
                n_related_posts
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
    from datetime import datetime
    from db import get_chronological_sample_posts

    parser = argparse.ArgumentParser(description="Process posts to compute metrics using the new class-based system")
    parser.add_argument("--every-n", type=int, default=50,
                       help="Sample every nth post chronologically (default: 50)")
    parser.add_argument("--version", type=str, default="v3_class_based",
                       help="Version ID for this processing run (default: v3_class_based)")
    parser.add_argument("--threads", type=int, default=4,
                       help="Number of worker threads (default: 4)")
    parser.add_argument("--model", type=str, default="gpt-5-mini",
                       choices=["gpt-5-nano", "gpt-5-mini", "gpt-5"],
                       help="Model to use (default: gpt-5-mini)")
    parser.add_argument("--related-posts", type=int, default=5,
                       help="Number of related posts to use for synthesis context (default: 5)")

    args = parser.parse_args()

    # All available V3 metrics (automatically includes synthesis metrics)
    metrics = [
        # "ValueV3",
        # "AuthorAuraV3",
        # "ReasoningQualityV3",
        # "CooperativenessV3",
        # "PrecisionV3",
        # "EmpiricalEvidenceQualityV3",
        # "ControversyTemperatureV3",
        # "MemeticPotentialV3",
        # "OverallEpistemicQualityV3",
        # "OverallKarmaPredictorV3"
        # "GptNanoOBOEpistemicQualityV0",
        # "GptMiniOBOEpistemicQualityV0",
        # "GptFullOBOEpistemicQualityV0",
        "GptNanoOBOQualityV0",
        "GptMiniOBOQualityV0",
        "GptFullOBOQualityV0",
    ]

    # Use fixed start date of 2024-01-01, no end date (latest available)
    start_datetime = datetime(2024, 1, 1)

    # Get posts using chronological sampling
    posts = get_chronological_sample_posts(
        n=args.every_n,
        start_datetime=start_datetime
    )

    print(f"Sampled {len(posts)} posts (every {args.every_n}th post from 2024-01-01)")
    if posts:
        print(f"Date range: {posts[0].posted_at} to {posts[-1].posted_at}")

    # Process posts
    process_posts(
        posts=posts,
        metrics=metrics,
        version_id=args.version,
        model=args.model,
        max_workers=args.threads,
        n_related_posts=args.related_posts
    )