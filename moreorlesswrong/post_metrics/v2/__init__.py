"""Epistemic virtue metrics v2 based on Paul's framework.

Primary Virtues:
- Truthfulness: Is the system providing truthful information?
- Value: Would the information make a difference if accurate?
- Cooperativeness: Does it improve the recipient's epistemic situation?

Derivative Virtues:
- Coherence: Are claims mutually consistent?
- Clarity: How easy is it to understand?
- Precision: How informative are the claims?
- Honesty: Is the system transparent and non-deceptive?

Other Metrics:
- Author Aura: EA fame and influence of the author
- External Validation: Empirical claim verification through web search
- Robustness: Two-step evaluation of post weaknesses and improvement potential
- Reasoning Quality: Evaluation of logical soundness and argument structure
"""

from .truthfulness_v2 import TruthfulnessV2, compute_truthfulness_v2
from .value_v2 import ValueV2, compute_value_v2
from .cooperativeness_v2 import CooperativenessV2, compute_cooperativeness_v2
from .coherence_v2 import CoherenceV2, compute_coherence_v2
from .clarity_v2 import ClarityV2, compute_clarity_v2
from .precision_v2 import PrecisionV2, compute_precision_v2
from .honesty_v2 import HonestyV2, compute_honesty_v2
from .author_aura_v2 import AuthorAuraV2, compute_author_aura_v2
from .external_validation_v2 import ExternalValidationV2, compute_external_validation_v2
from .robustness_v2 import RobustnessV2, compute_robustness_v2
from .reasoning_quality_v2 import ReasoningQualityV2, compute_reasoning_quality_v2
from .memetic_potential_v2 import MemeticPotentialV2, compute_memetic_potential_v2
from .title_clickability_v2 import TitleClickabilityV2, compute_title_clickability_v2
from .controversy_temperature_v2 import ControversyTemperatureV2, compute_controversy_temperature_v2
from .empirical_evidence_quality_v2 import EmpiricalEvidenceQualityV2, compute_empirical_evidence_quality_v2

__all__ = [
    # Primary virtues
    "TruthfulnessV2", "compute_truthfulness_v2",
    "ValueV2", "compute_value_v2", 
    "CooperativenessV2", "compute_cooperativeness_v2",
    # Derivative virtues
    "CoherenceV2", "compute_coherence_v2",
    "ClarityV2", "compute_clarity_v2",
    "PrecisionV2", "compute_precision_v2",
    "HonestyV2", "compute_honesty_v2",
    # Other metrics
    "AuthorAuraV2", "compute_author_aura_v2",
    "ExternalValidationV2", "compute_external_validation_v2",
    "RobustnessV2", "compute_robustness_v2",
    "ReasoningQualityV2", "compute_reasoning_quality_v2",
    "MemeticPotentialV2", "compute_memetic_potential_v2",
    "TitleClickabilityV2", "compute_title_clickability_v2",
    "ControversyTemperatureV2", "compute_controversy_temperature_v2",
    "EmpiricalEvidenceQualityV2", "compute_empirical_evidence_quality_v2",
]