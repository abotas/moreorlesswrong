"""Simplified class-based metric registry."""

from typing import Dict, List, Set, Type
from collections import defaultdict, deque

from metric_protocol import Metric, MetricContext, METRIC_REGISTRY
from models import Post


def get_all_metrics() -> List[str]:
    """Get list of all registered metric names."""
    return list(METRIC_REGISTRY.keys())


def get_standard_metrics() -> List[str]:
    """Get list of standard (non-synthesis) metric names."""
    return [name for name, cls in METRIC_REGISTRY.items() if not cls.dependencies()]


def get_synthesis_metrics() -> List[str]:
    """Get list of synthesis metric names."""
    return [name for name, cls in METRIC_REGISTRY.items() if cls.dependencies()]


def build_dependency_graph(requested_metrics: List[str]) -> Dict[str, Set[str]]:
    """
    Build a dependency graph for the requested metrics.
    Returns a dict mapping each metric to the set of metrics it depends on.
    """
    graph = defaultdict(set)
    to_process = deque(requested_metrics)
    processed = set()

    while to_process:
        metric_name = to_process.popleft()

        if metric_name in processed:
            continue

        if metric_name not in METRIC_REGISTRY:
            raise ValueError(f"Unknown metric: {metric_name}")

        metric_cls = METRIC_REGISTRY[metric_name]
        dependencies = metric_cls.dependencies()
        graph[metric_name] = set(dependencies)

        # Add dependencies to processing queue
        for dep in dependencies:
            if dep not in processed:
                to_process.append(dep)

        processed.add(metric_name)

    return dict(graph)


def topological_sort(graph: Dict[str, Set[str]]) -> List[str]:
    """
    Perform topological sort on the dependency graph.
    Returns metrics in order they should be computed.
    """
    # Count incoming edges
    in_degree = defaultdict(int)
    all_nodes = set()

    for node, deps in graph.items():
        all_nodes.add(node)
        for dep in deps:
            all_nodes.add(dep)
            in_degree[dep] += 0  # Ensure it exists
        in_degree[node] += 0  # Ensure it exists

    for node, deps in graph.items():
        for dep in deps:
            in_degree[node] += 1

    # Find nodes with no incoming edges
    queue = deque([node for node in all_nodes if in_degree[node] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        # Reduce in-degree for nodes that depend on this one
        for other_node, deps in graph.items():
            if node in deps:
                in_degree[other_node] -= 1
                if in_degree[other_node] == 0:
                    queue.append(other_node)

    if len(result) != len(all_nodes):
        raise ValueError("Circular dependency detected in metrics")

    return result


def compute_metrics(
    post: Post,
    requested_metrics: List[str],
    context: MetricContext
) -> Dict[str, Metric]:
    """
    Compute requested metrics with automatic dependency resolution.

    Args:
        post: The post to evaluate
        requested_metrics: List of metric names to compute
        context: Shared computation context

    Returns:
        Dictionary mapping metric names to computed metric objects
    """
    # Build dependency graph and get execution order
    dep_graph = build_dependency_graph(requested_metrics)
    execution_order = topological_sort(dep_graph)

    # Compute metrics in order
    computed_metrics: Dict[str, Metric] = {}
    errors: Dict[str, str] = {}

    for metric_name in execution_order:
        if metric_name not in METRIC_REGISTRY:
            errors[metric_name] = f"Metric {metric_name} not found in registry"
            continue

        metric_cls = METRIC_REGISTRY[metric_name]

        try:
            dependencies = metric_cls.dependencies()

            if dependencies:
                # Synthesis metric: pass required metrics as kwargs
                required_metrics = {}
                missing_deps = []

                for dep in dependencies:
                    if dep in computed_metrics:
                        required_metrics[dep] = computed_metrics[dep]
                    else:
                        missing_deps.append(dep)

                if missing_deps:
                    errors[metric_name] = f"Missing dependencies: {missing_deps}"
                    continue

                result = metric_cls.compute(post=post, context=context, **required_metrics)
            else:
                # Standard metric: just pass post and context
                result = metric_cls.compute(post=post, context=context)

            computed_metrics[metric_name] = result
            print(f"  ✓ Computed {metric_name}")

        except Exception as e:
            errors[metric_name] = str(e)
            print(f"  ✗ Error in {metric_name}: {e}")

    # Report any errors
    if errors:
        print(f"\nMetric computation errors:")
        for metric_name, error in errors.items():
            print(f"  - {metric_name}: {error}")

    # Filter to only requested metrics (exclude intermediate dependencies)
    return {
        name: metric
        for name, metric in computed_metrics.items()
        if name in requested_metrics
    }