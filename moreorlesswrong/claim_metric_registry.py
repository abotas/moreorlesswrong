from typing import List, Dict, Literal
from pydantic import BaseModel

from models import Post, Claim
from metrics.novelty import Novelty, compute_novelty
from metrics.novelty_support import NoveltySupport, compute_novelty_support
from metrics.inferential_support import InferentialSupport, compute_inferential_support
from metrics.external_validation import ExternalValidation, compute_external_validation
from metrics.robustness import Robustness, compute_robustness
from metrics.author_aura import AuthorAura, compute_author_aura
from metrics.value import Value, compute_value

# Registry of all available metrics - maps string name to compute function. should probably make this a decorator etc
METRIC_REGISTRY = {
    "Novelty": compute_novelty,
    "NoveltySupport": compute_novelty_support,
    "InferentialSupport": compute_inferential_support,
    "ExternalValidation": compute_external_validation,
    "Robustness": compute_robustness,
    "AuthorAura": compute_author_aura,
    "Value": compute_value
}


def compute_metrics_for_claim(
    metrics: List[str],
    claim: Claim,
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> Dict[str, BaseModel]:
    """Compute specified metrics for a claim.
    
    Args:
        metrics: List of metric names (strings) to compute
        claim: The claim to evaluate
        post: The post containing the claim
        model: The LLM model to use for computation
        
    Returns:
        Dictionary mapping metric name to computed metric object
    """
    results = {}
    
    for metric_name in metrics:
        if metric_name not in METRIC_REGISTRY:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        compute_fn = METRIC_REGISTRY[metric_name]
        metric_result = compute_fn(claim, post, model)
        results[metric_name] = metric_result
    
    return results