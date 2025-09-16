"""V3 metrics - Advanced metrics with sophisticated evaluation criteria."""

from .value_v3 import ValueV3
from .author_aura_v3 import AuthorAuraV3
from .reasoning_quality_v3 import ReasoningQualityV3
from .cooperativeness_v3 import CooperativenessV3
from .precision_v3 import PrecisionV3
from .empirical_evidence_quality_v3 import EmpiricalEvidenceQualityV3
from .controversy_temperature_v3 import ControversyTemperatureV3
from .memetic_potential_v3 import MemeticPotentialV3
from .overall_epistemic_quality_v3 import OverallEpistemicQualityV3
from .overall_karma_predictor_v3 import OverallKarmaPredictorV3

__all__ = [
    "ValueV3",
    "AuthorAuraV3",
    "ReasoningQualityV3",
    "CooperativenessV3",
    "PrecisionV3",
    "EmpiricalEvidenceQualityV3",
    "ControversyTemperatureV3",
    "MemeticPotentialV3",
    "OverallEpistemicQualityV3",
    "OverallKarmaPredictorV3",
]