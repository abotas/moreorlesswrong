"""Base class for synthesis metrics that combine multiple other metrics."""

from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel


class SynthesisMetric(BaseModel, ABC):
    """Base class for metrics that synthesize from other metrics.
    
    Synthesis metrics are computed from the outputs of other metrics
    rather than directly from post content. They run after regular
    metrics have been computed and saved.
    """
    
    @classmethod
    @abstractmethod
    def required_metrics(cls) -> List[str]:
        """Return list of metric names required as inputs.
        
        These should be the exact metric names as they appear
        in the metric registry (e.g., "ValueV3", "ReasoningQualityV3").
        """
        pass
    
    @classmethod
    def is_synthesis_metric(cls) -> bool:
        """Identify this as a synthesis metric."""
        return True
    
    @classmethod
    @abstractmethod
    def metric_name(cls) -> str:
        """Return the name of this metric."""
        pass
    
    @classmethod
    @abstractmethod
    def metric_score_fields(cls) -> List[str]:
        """Return list of score field names for this metric."""
        pass