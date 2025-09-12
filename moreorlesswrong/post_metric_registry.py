from typing import List, Dict, Literal
from pydantic import BaseModel

from models import Post

# V2 Metrics
from post_metrics.v2 import (
    TruthfulnessV2, compute_truthfulness_v2,
    ValueV2, compute_value_v2,
    CooperativenessV2, compute_cooperativeness_v2,
    CoherenceV2, compute_coherence_v2,
    ClarityV2, compute_clarity_v2,
    PrecisionV2, compute_precision_v2,
    HonestyV2, compute_honesty_v2,
    AuthorAuraV2, compute_author_aura_v2,
    ExternalValidationV2, compute_external_validation_v2,
    RobustnessV2, compute_robustness_v2,
    ReasoningQualityV2, compute_reasoning_quality_v2,
    MemeticPotentialV2, compute_memetic_potential_v2,
    TitleClickabilityV2, compute_title_clickability_v2,
    ControversyTemperatureV2, compute_controversy_temperature_v2,
    EmpiricalEvidenceQualityV2, compute_empirical_evidence_quality_v2
)

# V3 Metrics
from post_metrics.v3.value_v3 import ValueV3, compute_value_v3
from post_metrics.v3.author_aura_v3 import AuthorAuraV3, compute_author_aura_v3
from post_metrics.v3.reasoning_quality_v3 import ReasoningQualityV3, compute_reasoning_quality_v3
from post_metrics.v3.cooperativeness_v3 import CooperativenessV3, compute_cooperativeness_v3
from post_metrics.v3.precision_v3 import PrecisionV3, compute_precision_v3
from post_metrics.v3.empirical_evidence_quality_v3 import EmpiricalEvidenceQualityV3, compute_empirical_evidence_quality_v3

# Registry of all available post metrics
METRIC_CLASSES = {
    # # V1 Metrics
    # "PostNovelty": PostNovelty,
    # "PostNoveltySupport": PostNoveltySupport,
    # "PostInferentialSupport": PostInferentialSupport,
    # "PostEmpiricalClaimExternalValidation": PostEmpiricalClaimExternalValidation,
    # "PostRobustness": PostRobustness,
    # "PostAuthorAura": PostAuthorAura,
    # "PostValue": PostValue,
    # "PostClarity": PostClarity,
    # V2 Metrics
    "TruthfulnessV2": TruthfulnessV2,
    "ValueV2": ValueV2,
    "CooperativenessV2": CooperativenessV2,
    "CoherenceV2": CoherenceV2,
    "ClarityV2": ClarityV2,
    "PrecisionV2": PrecisionV2,
    "HonestyV2": HonestyV2,
    "AuthorAuraV2": AuthorAuraV2,
    "ExternalValidationV2": ExternalValidationV2,
    "RobustnessV2": RobustnessV2,
    "ReasoningQualityV2": ReasoningQualityV2,
    "MemeticPotentialV2": MemeticPotentialV2,
    "TitleClickabilityV2": TitleClickabilityV2,
    "ControversyTemperatureV2": ControversyTemperatureV2,
    "EmpiricalEvidenceQualityV2": EmpiricalEvidenceQualityV2,
    # V3 Metrics
    "ValueV3": ValueV3,
    "AuthorAuraV3": AuthorAuraV3,
    "ReasoningQualityV3": ReasoningQualityV3,
    "CooperativenessV3": CooperativenessV3,
    "PrecisionV3": PrecisionV3,
    "EmpiricalEvidenceQualityV3": EmpiricalEvidenceQualityV3
}
POST_METRIC_REGISTRY = {
    # V1 Metrics
    # "PostNovelty": compute_post_novelty,
    # "PostNoveltySupport": compute_post_novelty_support,
    # "PostInferentialSupport": compute_post_inferential_support,
    # "PostEmpiricalClaimExternalValidation": compute_post_empirical_claim_external_validation,
    # "PostRobustness": compute_post_robustness,
    # "PostAuthorAura": compute_post_author_aura,
    # "PostValue": compute_post_value,
    # "PostClarity": compute_post_clarity,
    # V2 Metrics
    "TruthfulnessV2": compute_truthfulness_v2,
    "ValueV2": compute_value_v2,
    "CooperativenessV2": compute_cooperativeness_v2,
    "CoherenceV2": compute_coherence_v2,
    "ClarityV2": compute_clarity_v2,
    "PrecisionV2": compute_precision_v2,
    "HonestyV2": compute_honesty_v2,
    "AuthorAuraV2": compute_author_aura_v2,
    "ExternalValidationV2": compute_external_validation_v2,
    "RobustnessV2": compute_robustness_v2,
    "ReasoningQualityV2": compute_reasoning_quality_v2,
    "MemeticPotentialV2": compute_memetic_potential_v2,
    "TitleClickabilityV2": compute_title_clickability_v2,
    "ControversyTemperatureV2": compute_controversy_temperature_v2,
    "EmpiricalEvidenceQualityV2": compute_empirical_evidence_quality_v2,
    # V3 Metrics
    "ValueV3": compute_value_v3,
    "AuthorAuraV3": compute_author_aura_v3,
    "ReasoningQualityV3": compute_reasoning_quality_v3,
    "CooperativenessV3": compute_cooperativeness_v3,
    "PrecisionV3": compute_precision_v3,
    "EmpiricalEvidenceQualityV3": compute_empirical_evidence_quality_v3
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