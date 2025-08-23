"""Registry of all metric models for use in streamlit and other components."""

from typing import Dict, Type
from pydantic import BaseModel

from metrics.novelty import Novelty
from metrics.inferential_support import InferentialSupport
from metrics.external_validation import ExternalValidation
from metrics.robustness import Robustness
from metrics.author_aura import AuthorAura

# Map metric names to their model classes
METRIC_MODELS: Dict[str, Type[BaseModel]] = {
    "Novelty": Novelty,
    "InferentialSupport": InferentialSupport,
    "ExternalValidation": ExternalValidation,
    "Robustness": Robustness,
    "AuthorAura": AuthorAura
}

def get_metric_score_fields(metric_name: str) -> list[str]:
    """Get the score field names for a given metric."""
    if metric_name not in METRIC_MODELS:
        return []
    return METRIC_MODELS[metric_name].metric_score_fields()