"""Protocol definitions for the metric system."""

from typing import Protocol, List, Dict, Any, runtime_checkable, Optional, Type
from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass
from models import Post


class MetricContext(BaseModel):
    """Shared context for all metric computations."""
    model: str
    n_related_posts: int


# Global registry populated by metaclass
METRIC_REGISTRY: Dict[str, Type['Metric']] = {}


class MetricMeta(ModelMetaclass):
    """Metaclass that inherits from Pydantic's ModelMetaclass and adds auto-registration."""

    def __new__(cls, name, bases, namespace, **kwargs):
        metric_cls = super().__new__(cls, name, bases, namespace, **kwargs)

        # Register metrics that have a compute method (excluding base class)
        if (hasattr(metric_cls, 'compute') and
            name != 'Metric' and
            hasattr(metric_cls, 'metric_name')):  # Ensure it's a proper metric class
            METRIC_REGISTRY[name] = metric_cls
            print(f"Auto-registered metric: {name}")

        return metric_cls


class Metric(BaseModel, metaclass=MetricMeta):
    """Base class for all metrics with auto-registration."""

    @classmethod
    def compute(cls, post: Post, context: MetricContext, **deps) -> 'Metric':
        """Compute this metric. Override in subclasses."""
        raise NotImplementedError(f"Metric {cls.__name__} must implement compute()")

    @classmethod
    def dependencies(cls) -> List[str]:
        """Return list of metrics this one depends on. Override in synthesis metrics."""
        return []

    @classmethod
    def metric_name(cls) -> str:
        """Return the name of this metric."""
        return cls.__name__

    @classmethod
    def metric_score_fields(cls) -> List[str]:
        """Return list of score field names. Override in subclasses."""
        return []

    @classmethod
    def human_readable_names(cls) -> Dict[str, str]:
        """Return human-readable names for fields. Override in subclasses."""
        return {}


@runtime_checkable
class MetricProtocol(Protocol):
    """Protocol that all metric classes must follow."""

    @classmethod
    def metric_name(cls) -> str:
        """Return the name of this metric."""
        ...

    @classmethod
    def metric_score_fields(cls) -> List[str]:
        """Return list of score field names."""
        ...

    @classmethod
    def human_readable_names(cls) -> Dict[str, str]:
        """Return human-readable names for fields."""
        ...


class StandardMetric(MetricProtocol):
    """Protocol for metrics computed directly from posts."""
    pass


class SynthesisMetric(MetricProtocol):
    """Protocol for metrics computed from other metrics."""

    @classmethod
    def required_metrics(cls) -> List[str]:
        """Return list of metrics required as inputs."""
        ...


@runtime_checkable
class ComputeFunction(Protocol):
    """Protocol for metric compute functions."""

    def __call__(self, post: Post, context: MetricContext, **metrics: Any) -> BaseModel:
        """
        Compute the metric.

        Args:
            post: The post to evaluate
            context: Shared computation context
            **metrics: For synthesis metrics, the required input metrics

        Returns:
            The computed metric object
        """
        ...


class MetricRegistration:
    """Registration info for a metric."""

    def __init__(
        self,
        name: str,
        compute_fn: ComputeFunction,
        output_class: Type[BaseModel],
        dependencies: Optional[List[str]] = None
    ):
        self.name = name
        self.compute_fn = compute_fn
        self.output_class = output_class
        self.dependencies = dependencies or []
        self.is_synthesis = bool(dependencies)

    def __repr__(self) -> str:
        deps = f", deps={self.dependencies}" if self.dependencies else ""
        return f"MetricRegistration({self.name}{deps})"