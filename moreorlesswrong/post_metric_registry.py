from typing import List, Dict, Literal
from pydantic import BaseModel

from models import Post

# V2 Metrics
from post_metrics.v2 import (
    TitleClickabilityV2, compute_title_clickability_v2,
)

# V3 Metrics
from post_metrics.v3.value_v3 import ValueV3, compute_value_v3
from post_metrics.v3.author_aura_v3 import AuthorAuraV3, compute_author_aura_v3
from post_metrics.v3.reasoning_quality_v3 import ReasoningQualityV3, compute_reasoning_quality_v3
from post_metrics.v3.cooperativeness_v3 import CooperativenessV3, compute_cooperativeness_v3
from post_metrics.v3.precision_v3 import PrecisionV3, compute_precision_v3
from post_metrics.v3.empirical_evidence_quality_v3 import EmpiricalEvidenceQualityV3, compute_empirical_evidence_quality_v3
from post_metrics.v3.controversy_temperature_v3 import ControversyTemperatureV3, compute_controversy_temperature_v3
# from post_metrics.v3.robustness_v3 import RobustnessV3, compute_robustness_v3
from post_metrics.v3.memetic_potential_v3 import MemeticPotentialV3, compute_memetic_potential_v3
from post_metrics.v3.overall_epistemic_quality_v3 import OverallEpistemicQualityV3
from post_metrics.v3.overall_karma_predictor_v3 import OverallKarmaPredictorV3

# V0 Metrics (Simple holistic evaluations)
from post_metrics.v0 import (
    GptNanoOBOEpistemicQualityV0, compute_gpt_nano_obo_epistemic_quality_v0,
    GptMiniOBOEpistemicQualityV0, compute_gpt_mini_obo_epistemic_quality_v0,
    GptFullOBOEpistemicQualityV0, compute_gpt_full_obo_epistemic_quality_v0
)

# Registry of all available post metrics
METRIC_CLASSES = {
    # V0 Metrics
    "GptNanoOBOEpistemicQualityV0": GptNanoOBOEpistemicQualityV0,
    "GptMiniOBOEpistemicQualityV0": GptMiniOBOEpistemicQualityV0,
    "GptFullOBOEpistemicQualityV0": GptFullOBOEpistemicQualityV0,
    # V3 Metrics
    "ValueV3": ValueV3,
    "AuthorAuraV3": AuthorAuraV3,
    "ReasoningQualityV3": ReasoningQualityV3,
    "CooperativenessV3": CooperativenessV3,
    "PrecisionV3": PrecisionV3,
    "EmpiricalEvidenceQualityV3": EmpiricalEvidenceQualityV3,
    "ControversyTemperatureV3": ControversyTemperatureV3,
    # "RobustnessV3": RobustnessV3,
    "MemeticPotentialV3": MemeticPotentialV3,
    # Synthesis Metrics (not in POST_METRIC_REGISTRY as they're computed separately)
    "OverallEpistemicQualityV3": OverallEpistemicQualityV3,
    "OverallKarmaPredictorV3": OverallKarmaPredictorV3
}
POST_METRIC_REGISTRY = {
    # V0 Metrics
    "GptNanoOBOEpistemicQualityV0": compute_gpt_nano_obo_epistemic_quality_v0,
    "GptMiniOBOEpistemicQualityV0": compute_gpt_mini_obo_epistemic_quality_v0,
    "GptFullOBOEpistemicQualityV0": compute_gpt_full_obo_epistemic_quality_v0,
    
    # V2 Metrics
    "TitleClickabilityV2": compute_title_clickability_v2,

    # V3 Metrics
    "ValueV3": compute_value_v3,
    "AuthorAuraV3": compute_author_aura_v3,
    "ReasoningQualityV3": compute_reasoning_quality_v3,
    "CooperativenessV3": compute_cooperativeness_v3,
    "PrecisionV3": compute_precision_v3,
    "EmpiricalEvidenceQualityV3": compute_empirical_evidence_quality_v3,
    "ControversyTemperatureV3": compute_controversy_temperature_v3,
    # "RobustnessV3": compute_robustness_v3,
    "MemeticPotentialV3": compute_memetic_potential_v3
}


def compute_metrics_for_post(
    metrics: List[str],
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini",
    bypass_synthesizer: bool = False,
    n_related_posts: int = 5,
) -> List[BaseModel | tuple[str, str]]:
    """Compute specified metrics for a post.
    
    Args:
        metrics: List of metric names (strings) to compute
        post: The post to evaluate
        model: The LLM model to use for computation
        bypass_synthesizer: Whether to bypass synthesizer and use raw related posts (default: False)
        
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
            # V3 metrics support bypass_synthesizer parameter
            if metric_name.endswith("V3"):
                metric_result = compute_fn(post, model, bypass_synthesizer=bypass_synthesizer, n_related_posts=n_related_posts)
            else:
                # V2 and other metrics use the original signature
                metric_result = compute_fn(post, model)
            results.append(metric_result)
        except Exception as e:
            # Append tuple with metric name and error for failed metrics
            results.append((metric_name, str(e)))
            print(f"    ERROR in {metric_name}: {e}")
            
            # For pydantic validation errors, show more detail
            from pydantic import ValidationError
            if isinstance(e, ValidationError):
                print(f"    PYDANTIC VALIDATION DETAILS for {metric_name}:")
                for error in e.errors():
                    field = error.get('loc', ['unknown'])[0] if error.get('loc') else 'unknown'
                    input_val = error.get('input')
                    expected_type = error.get('type', 'unknown')
                    print(f"      Field '{field}': Expected {expected_type}, got {type(input_val).__name__}: {repr(input_val)}")
                    if isinstance(input_val, (list, dict)) and len(str(input_val)) > 200:
                        print(f"      Field '{field}' preview: {repr(str(input_val)[:200])}...")
                    
            # For JSON parsing errors, try to show the raw content that failed
            if "json" in str(e).lower() or "parse" in str(e).lower():
                print(f"    This might be a JSON parsing issue for {metric_name}")
                print(f"    Try checking the LLM response format")
    
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