"""Registry for synthesis metrics that combine outputs from other metrics."""

from typing import Dict, Any, Type, Callable
from pydantic import BaseModel

from synthesis_metric_base import SynthesisMetric

# Import synthesis metrics
from post_metrics.v3.overall_epistemic_quality_v3 import (
    OverallEpistemicQualityV3,
    compute_overall_epistemic_quality_v3
)
from post_metrics.v3.overall_karma_predictor_v3 import (
    OverallKarmaPredictorV3,
    compute_overall_karma_predictor_v3
)

# Import the individual metric classes we need to reconstruct
from post_metrics.v3.value_v3 import ValueV3
from post_metrics.v3.reasoning_quality_v3 import ReasoningQualityV3
from post_metrics.v3.cooperativeness_v3 import CooperativenessV3
from post_metrics.v3.precision_v3 import PrecisionV3
from post_metrics.v3.empirical_evidence_quality_v3 import EmpiricalEvidenceQualityV3
from post_metrics.v3.memetic_potential_v3 import MemeticPotentialV3
from post_metrics.v3.author_aura_v3 import AuthorAuraV3
from post_metrics.v3.controversy_temperature_v3 import ControversyTemperatureV3


# Registry of synthesis metric classes
SYNTHESIS_METRIC_CLASSES: Dict[str, Type[SynthesisMetric]] = {
    "OverallEpistemicQualityV3": OverallEpistemicQualityV3,
    "OverallKarmaPredictorV3": OverallKarmaPredictorV3
}

# Registry of synthesis metric compute functions
SYNTHESIS_METRICS: Dict[str, Callable] = {
    "OverallEpistemicQualityV3": compute_overall_epistemic_quality_v3,
    "OverallKarmaPredictorV3": compute_overall_karma_predictor_v3
}

# Registry for reconstructing metric objects from JSON
METRIC_RECONSTRUCTORS: Dict[str, Type[BaseModel]] = {
    "ValueV3": ValueV3,
    "ReasoningQualityV3": ReasoningQualityV3,
    "CooperativenessV3": CooperativenessV3,
    "PrecisionV3": PrecisionV3,
    "EmpiricalEvidenceQualityV3": EmpiricalEvidenceQualityV3,
    "MemeticPotentialV3": MemeticPotentialV3,
    "AuthorAuraV3": AuthorAuraV3,
    "ControversyTemperatureV3": ControversyTemperatureV3,
    "OverallEpistemicQualityV3": OverallEpistemicQualityV3,
    "OverallKarmaPredictorV3": OverallKarmaPredictorV3
}


def convert_metric_name_to_param(metric_name: str) -> str:
    """Convert metric class name to parameter name.
    
    Examples:
        "ValueV3" -> "value_v3"
        "ReasoningQualityV3" -> "reasoning_quality_v3"
        "EmpiricalEvidenceQualityV3" -> "empirical_evidence_quality_v3"
    """
    # Remove V3 suffix
    base_name = metric_name.replace("V3", "")
    
    # Convert camelCase to snake_case
    import re
    # Insert underscores before uppercase letters (except the first one)
    snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', base_name).lower()
    
    # Add _v3 suffix
    return snake_case + "_v3"


def reconstruct_metric(metric_data: Dict[str, Any], metric_name: str) -> BaseModel:
    """Reconstruct a metric object from its JSON representation.
    
    Args:
        metric_data: The metric's data as a dictionary (from JSON)
        metric_name: The name of the metric (e.g., "ValueV3")
        
    Returns:
        The reconstructed metric object
        
    Raises:
        ValueError: If the metric name is not in the reconstructor registry
    """
    if metric_name not in METRIC_RECONSTRUCTORS:
        raise ValueError(f"No reconstructor found for metric: {metric_name}")
    
    metric_class = METRIC_RECONSTRUCTORS[metric_name]
    return metric_class(**metric_data)


def load_v3_metrics_from_json(existing_metrics: Dict[str, Any]) -> Dict[str, BaseModel]:
    """Load all V3 metrics from JSON data into pydantic objects.
    
    Args:
        existing_metrics: Dictionary loaded from JSON file containing all metrics
        
    Returns:
        Dictionary mapping metric names to reconstructed pydantic objects (only V3 metrics)
    """
    v3_metrics = {}
    
    for metric_name, metric_data in existing_metrics.items():
        # Only process V3 metrics
        if "V3" in metric_name and isinstance(metric_data, dict):
            try:
                metric_obj = reconstruct_metric(metric_data, metric_name)
                # Convert metric name to parameter name (e.g., "ReasoningQualityV3" -> "reasoning_quality_v3")
                param_name = convert_metric_name_to_param(metric_name)
                v3_metrics[param_name] = metric_obj
            except Exception as e:
                print(f"    WARNING: Failed to reconstruct {metric_name}: {e}")
                continue
    
    return v3_metrics