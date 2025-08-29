from typing import List, Dict, Literal
from pydantic import BaseModel

from models import Post
from post_metrics.novelty import PostNovelty, compute_post_novelty
from post_metrics.novelty_support import PostNoveltySupport, compute_post_novelty_support
from post_metrics.inferential_support import PostInferentialSupport, compute_post_inferential_support
from post_metrics.external_validation import PostExternalValidation, compute_post_external_validation
from post_metrics.robustness import PostRobustness, compute_post_robustness
from post_metrics.author_aura import PostAuthorAura, compute_post_author_aura
from post_metrics.value import PostValue, compute_post_value
from post_metrics.clarity import PostClarity, compute_post_clarity

# Registry of all available post metrics
METRIC_CLASSES = {
    "PostNovelty": PostNovelty,
    "PostNoveltySupport": PostNoveltySupport,
    "PostInferentialSupport": PostInferentialSupport,
    "PostExternalValidation": PostExternalValidation,
    "PostRobustness": PostRobustness,
    "PostAuthorAura": PostAuthorAura,
    "PostValue": PostValue,
    "PostClarity": PostClarity
}
POST_METRIC_REGISTRY = {
    "PostNovelty": compute_post_novelty,
    "PostNoveltySupport": compute_post_novelty_support,
    "PostInferentialSupport": compute_post_inferential_support,
    "PostExternalValidation": compute_post_external_validation,
    "PostRobustness": compute_post_robustness,
    "PostAuthorAura": compute_post_author_aura,
    "PostValue": compute_post_value,
    "PostClarity": compute_post_clarity
}


def compute_metrics_for_post(
    metrics: List[str],
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> List[BaseModel | tuple[str, str]]:
    """Compute specified metrics for a post.
    
    Args:
        metrics: List of metric names (strings) to compute
        post: The post to evaluate
        model: The LLM model to use for computation
        
    Returns:
        List containing either:
        - Computed metric objects (on success)
        - Tuples of (metric_name, error_string) (on failure)
    """
    results = []
    
    for metric_name in metrics:
        if metric_name not in POST_METRIC_REGISTRY:
            raise ValueError(f"Unknown post metric: {metric_name}")
        
        compute_fn = POST_METRIC_REGISTRY[metric_name]
        try:
            metric_result = compute_fn(post, model)
            results.append(metric_result)
        except Exception as e:
            # Append tuple with metric name and error for failed metrics
            results.append((metric_name, str(e)))
            print(f"    ERROR in {metric_name}: {e}")
    
    return results


def get_human_readable_name(metric_field: str) -> str:
    """Convert a metric field name to human readable format."""
    # Remove Post prefix and find the metric class
    for metric_name, metric_class in METRIC_CLASSES.items():
        if metric_field.startswith(metric_name + "_"):
            field_name = metric_field[len(metric_name) + 1:]  # Remove "PostMetricName_"
            if hasattr(metric_class, 'human_readable_names'):
                readable_names = metric_class.human_readable_names()
                if field_name in readable_names:
                    return readable_names[field_name]
    
    # Fallback: convert underscore to spaces and title case
    return metric_field.replace('_', ' ').title()