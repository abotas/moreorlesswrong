import json
from pathlib import Path
from typing import List, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

from models import Post
from post_metric_registry import compute_metrics_for_post
from synthesis_metric_registry import (
    SYNTHESIS_METRICS, 
    SYNTHESIS_METRIC_CLASSES,
    load_v3_metrics_from_json,
    convert_metric_name_to_param
)


def process_single_post(
    post: Post,
    metrics: List[str],
    version_id: str,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"],
    synthesis_metrics: List[str] = None,
    n_related_posts: int = 5,
    bypass_synthesizer: bool = False
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
        computed = compute_metrics_for_post(missing_metrics, post, model, bypass_synthesizer, n_related_posts)
        
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
    
    # Phase 2: Compute synthesis metrics if dependencies are met
    # This runs after individual metrics are saved to disk
    synthesis_computed = []
    
    # Only process synthesis metrics if they were requested
    if synthesis_metrics:
        # Load all V3 metrics into pydantic objects once
        try:
            v3_metric_objects = load_v3_metrics_from_json(existing_metrics)
            print(f"  Loaded {len(v3_metric_objects)} V3 metric objects for synthesis")
        except Exception as e:
            print(f"  ERROR loading V3 metrics for synthesis: {e}")
            v3_metric_objects = {}
        
        for metric_name in synthesis_metrics:
            # Skip if already computed
            if metric_name in existing_metrics:
                continue

            compute_fn = SYNTHESIS_METRICS[metric_name]
            metric_class = SYNTHESIS_METRIC_CLASSES[metric_name]
            required = metric_class.required_metrics()
        
            # Check if all required metrics exist in our loaded V3 objects
            required_param_names = [
                convert_metric_name_to_param(req_metric) 
                for req_metric in required
            ]
            
            if all(param_name in v3_metric_objects for param_name in required_param_names):
                print(f"  Computing synthesis metric: {metric_name}")
                try:
                    # Prepare inputs using pre-loaded V3 metric objects
                    inputs = {param_name: v3_metric_objects[param_name] for param_name in required_param_names}
                    
                    # Add post, model, and n_related_posts parameters
                    inputs['post'] = post
                    inputs['model'] = model
                    inputs['n_related_posts'] = n_related_posts
                    
                    # Compute synthesis metric
                    result = compute_fn(**inputs)
                    existing_metrics[metric_name] = result.model_dump()
                    synthesis_computed.append(metric_name)
                    
                except Exception as e:
                    print(f"    ERROR in synthesis metric {metric_name}: {e}")
            else:
                missing = [param for param in required_param_names if param not in v3_metric_objects]
                print(f"  Skipping synthesis metric {metric_name} because required metrics are missing: {missing}")
    
    # Save updated metrics if any synthesis metrics were computed
    if synthesis_computed:
        with open(metrics_file, 'w') as f:
            json.dump(existing_metrics, f, indent=2)
        print(f"  Computed {len(synthesis_computed)} synthesis metrics: {', '.join(synthesis_computed)}")
    
    return post.post_id


def process_posts(
    posts: List[Post],
    metrics: List[str],
    version_id: str,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini",
    max_workers: int = 4,
    synthesis_metrics: List[str] = None,
    n_related_posts: int = 5,
    bypass_synthesizer: bool = False
):
    """Process posts to compute metrics without extracting claims.
    
    This function is interruptable and will skip already processed data.
    
    Args:
        posts: List of posts to process
        metrics: List of metric names (strings) to compute
        version_id: Version identifier for this processing run
        model: LLM model to use
        max_workers: Maximum number of concurrent threads (default: 4)
        synthesis_metrics: List of synthesis metric names to compute (optional)
        n_related_posts: Number of related posts to use for synthesis context (default: 5)
        bypass_synthesizer: Whether to bypass synthesizer and use raw related posts (default: False)
    """
    print(f"\n{'='*60}")
    print(f"Starting post-level pipeline with {max_workers} worker threads")
    print(f"Version: {version_id}")
    print(f"Model: {model}")
    print(f"Posts to process: {len(posts)}")
    print(f"Metrics ({len(metrics)}): {', '.join(metrics)}")
    if synthesis_metrics:
        print(f"Synthesis Metrics ({len(synthesis_metrics)}): {', '.join(synthesis_metrics)}")
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
                synthesis_metrics,
                n_related_posts,
                bypass_synthesizer
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
    
    parser = argparse.ArgumentParser(description="Process posts to compute metrics using chronological sampling")
    parser.add_argument("--every-n", type=int, default=50,
                       help="Sample every nth post chronologically (default: 50)")
    parser.add_argument("--version", type=str, default="post_v2",
                       help="Version ID for this processing run (default: post_v2)")
    parser.add_argument("--threads", type=int, default=4,
                       help="Number of worker threads (default: 4)")
    parser.add_argument("--model", type=str, default="gpt-5-mini",
                       choices=["gpt-5-nano", "gpt-5-mini", "gpt-5"],
                       help="Model to use (default: gpt-5-mini)")
    parser.add_argument("--related-posts", type=int, default=5,
                       help="Number of related posts to use for synthesis context (default: 5)")
    parser.add_argument("--bypass-synthesizer", action="store_true",
                       help="Bypass synthesizer agent and use raw 20k char previews of related posts")
    
    args = parser.parse_args()
    
    # Hardcoded list of metrics to compute
    metrics = [
        # V2 Primary Virtues from paul's framework
        # "TruthfulnessV2",
        "ValueV2", 
        "CooperativenessV2",
        # V2 Derivative Virtues from paul's framework
        # "CoherenceV2",
        "ClarityV2",
        "PrecisionV2",
        # "HonestyV2",
        # Metrics that were correlative to post karma in V1
        "AuthorAuraV2",
        "ExternalValidationV2",
        "RobustnessV2",
        "ReasoningQualityV2",
        # New engagement/karma-predictive metrics
        # "MemeticPotentialV2",
        "TitleClickabilityV2", 
        "ControversyTemperatureV2",
        "EmpiricalEvidenceQualityV2",
        # V3 Metrics with synthesized context
        "ValueV3",
        "AuthorAuraV3", 
        "ReasoningQualityV3",
        "CooperativenessV3",
        "PrecisionV3",
        "EmpiricalEvidenceQualityV3",
        "ControversyTemperatureV3",
        "MemeticPotentialV3"
    ]
    
    # Synthesis metrics to compute (after individual metrics)
    synthesis_metrics = [
        "OverallEpistemicQualityV3",
        "OverallKarmaPredictorV3"
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
        synthesis_metrics=synthesis_metrics,
        n_related_posts=args.related_posts,
        bypass_synthesizer=args.bypass_synthesizer
    )