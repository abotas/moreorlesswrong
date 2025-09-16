#!/usr/bin/env python3
"""
Generic synthetic post grader that can process any synthetic post through the v3 metrics pipeline.
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from models import Post
from post_metric_pipeline import process_single_post
from typing import Dict, Any, Optional


def create_post_from_synthetic_data(
    synthetic_data: Dict[str, Any],
    post_id_suffix: str = ""
) -> Post:
    """Convert synthetic post data into a Post model."""

    base_post_id = synthetic_data.get("post_id", "synthetic_post")
    if post_id_suffix:
        post_id = f"{base_post_id}_{post_id_suffix}"
    else:
        post_id = base_post_id

    return Post(
        id=synthetic_data.get("id", 999999),
        post_id=post_id,
        title=synthetic_data["title"],
        title_normalized=synthetic_data["title"].lower(),
        page_url=synthetic_data.get("page_url", f"https://synthetic.ea.forum/post/{post_id}"),
        html_body=f"<div>{synthetic_data['content']}</div>",
        base_score=synthetic_data.get("karma", 100),
        comment_count=synthetic_data.get("comment_count", 10),
        posted_at=datetime.fromisoformat(synthetic_data.get("timestamp", "2025-01-15T12:00:00Z").replace('Z', '+00:00')),
        created_at=datetime.fromisoformat(synthetic_data.get("timestamp", "2025-01-15T12:00:00Z").replace('Z', '+00:00')),
        author_id=synthetic_data.get("author_id", "synthetic_author"),
        author_display_name=synthetic_data.get("author", "Synthetic Author"),
        coauthor_ids=None,
        coauthor_names=None,
        tag_ids=None,
        tag_names=synthetic_data.get("tags", ["Meta"]),
        markdown_content=synthetic_data["content"],
        word_count=synthetic_data.get("word_count", len(synthetic_data["content"].split())),
        reading_time_minutes=synthetic_data.get("word_count", len(synthetic_data["content"].split())) // 200,
        external_links=[],
        short_summary=synthetic_data.get("short_summary", synthetic_data["content"][:200] + "..."),
        long_summary=synthetic_data.get("long_summary", synthetic_data["content"][:500] + "..."),
        source_type="Synthetic",
        processing_version="1.0",
        processing_errors=None,
        scraped_at=datetime.now()
    )


def load_synthetic_post(file_path: str) -> Dict[str, Any]:
    """Load synthetic post data from a Python file or JSON file."""

    file_path = Path(file_path)

    if file_path.suffix == ".json":
        with open(file_path, 'r') as f:
            return json.load(f)

    elif file_path.suffix == ".py":
        # Import the Python file as a module
        import importlib.util
        spec = importlib.util.spec_from_file_location("synthetic_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Look for synthetic post variables
        for attr_name in dir(module):
            if attr_name.startswith("synthetic") and isinstance(getattr(module, attr_name), dict):
                return getattr(module, attr_name)

        raise ValueError(f"No synthetic post data found in {file_path}")

    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def grade_synthetic_post(
    synthetic_file: str,
    model: str = "gpt-5-mini",
    version_id: str = "synthetic_v3",
    output_suffix: str = "",
    metrics: Optional[list] = None,
    synthesis_metrics: Optional[list] = None
) -> str:
    """
    Grade a synthetic post through the v3 metrics pipeline.

    Args:
        synthetic_file: Path to synthetic post file (.py or .json)
        model: Model to use for grading (default: gpt-5-mini)
        version_id: Version ID for metrics storage
        output_suffix: Suffix to add to output filename
        metrics: List of v3 metrics to compute (default: all standard metrics)
        synthesis_metrics: List of synthesis metrics to compute

    Returns:
        Path to the saved metrics file
    """

    # Default metrics if not specified
    if metrics is None:
        metrics = [
            "ValueV3",
            "AuthorAuraV3",
            "ReasoningQualityV3",
            "CooperativenessV3",
            "PrecisionV3",
            "EmpiricalEvidenceQualityV3",
            "ControversyTemperatureV3",
            "MemeticPotentialV3"
        ]

    if synthesis_metrics is None:
        synthesis_metrics = [
            "OverallEpistemicQualityV3",
            "OverallKarmaPredictorV3"
        ]

    print("=" * 60)
    print(f"Grading synthetic post with {model}")
    print("=" * 60)

    # Load synthetic post data
    print(f"Loading synthetic post from: {synthetic_file}")
    synthetic_data = load_synthetic_post(synthetic_file)

    # Create post object
    post = create_post_from_synthetic_data(synthetic_data, output_suffix)

    print(f"Post: {post.title}")
    print(f"Author: {post.author_display_name}")
    print(f"Word count: {post.word_count}")

    if "cluster_name" in synthetic_data:
        print(f"Cluster: {synthetic_data['cluster_name']} (ID: {synthetic_data.get('cluster_id', 'N/A')})")

    print(f"\n🤖 Using model: {model}")
    print(f"Running {len(metrics)} individual v3 metrics...")
    print(f"Running {len(synthesis_metrics)} synthesis metrics...")

    # Process through pipeline
    try:
        process_single_post(
            post=post,
            metrics=metrics,
            version_id=version_id,
            model=model,
            synthesis_metrics=synthesis_metrics,
            n_related_posts=5,
            bypass_synthesizer=True
        )

        # Determine output file path
        metrics_file = f"../data/post_metrics/{version_id}/{post.post_id}.json"

        print(f"\n✅ Successfully processed synthetic post!")
        print(f"📁 Metrics saved to: {metrics_file}")

        # Display results summary
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                results = json.load(f)

            print(f"\n📊 {model.upper()} RESULTS SUMMARY:")
            print("=" * 50)

            key_metrics = {
                "ValueV3": "value_score",
                "ReasoningQualityV3": "reasoning_quality_score",
                "CooperativenessV3": "cooperativeness_score",
                "PrecisionV3": "precision_score",
                "EmpiricalEvidenceQualityV3": "empirical_evidence_quality_score",
                "ControversyTemperatureV3": "controversy_temperature_score",
                "MemeticPotentialV3": "memetic_potential_score",
                "OverallEpistemicQualityV3": "overall_epistemic_quality_score",
                "OverallKarmaPredictorV3": "predicted_karma_score"
            }

            for metric_class, score_field in key_metrics.items():
                if metric_class in results:
                    score = results[metric_class].get(score_field, "N/A")
                    metric_display = metric_class.replace("V3", "").ljust(25)
                    print(f"{metric_display}: {score}")

            print("=" * 50)

        return metrics_file

    except Exception as e:
        print(f"\n❌ Error processing synthetic post: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Command line interface for the synthetic post grader."""

    parser = argparse.ArgumentParser(description="Grade synthetic posts through v3 metrics pipeline")
    parser.add_argument("synthetic_file", help="Path to synthetic post file (.py or .json)")
    parser.add_argument("--model", default="gpt-5-mini", help="Model to use for grading")
    parser.add_argument("--version-id", default="synthetic_v3", help="Version ID for metrics storage")
    parser.add_argument("--output-suffix", default="", help="Suffix to add to output filename")
    parser.add_argument("--metrics", nargs="*", help="Specific v3 metrics to compute")
    parser.add_argument("--synthesis-metrics", nargs="*", help="Specific synthesis metrics to compute")

    args = parser.parse_args()

    try:
        metrics_file = grade_synthetic_post(
            synthetic_file=args.synthetic_file,
            model=args.model,
            version_id=args.version_id,
            output_suffix=args.output_suffix,
            metrics=args.metrics,
            synthesis_metrics=args.synthesis_metrics
        )

        print(f"\n🎉 Grading complete! Results saved to: {metrics_file}")

    except Exception as e:
        print(f"\n💥 Grading failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()