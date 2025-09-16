from typing import Literal
from pydantic import BaseModel
import json

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair
from db import get_n_most_recent_posts_in_same_cluster
from prev_post_synthesizer import synthesize_context
from raw_context_formatter import format_raw_related_posts
from metric_protocol import Metric, MetricContext


class ReasoningQualityV3(Metric):
    post_id: str
    reasoning_quality_score: int  # 1-10 reasoning quality score
    thesis: str  # The identified main thesis
    logical_arguments: str  # Key logical arguments identified
    explanation: str  # How effectively arguments support thesis
    
    @classmethod
    def metric_name(cls) -> str:
        return "ReasoningQualityV3"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["reasoning_quality_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "reasoning_quality_score": "Reasoning Quality Score"
        }

    @classmethod
    def compute(cls, post: Post, context: MetricContext) -> "ReasoningQualityV3":
        """Compute reasoning quality score for a post with synthesized information from related posts.

        Args:
            post: The post to evaluate
            context: Shared metric computation context

        Returns:
            ReasoningQualityV3 metric object
        """
        post_text = post.markdown_content or post.html_body or ""

        related_posts = get_n_most_recent_posts_in_same_cluster(
            post_id=post.post_id,
            cluster_cardinality=12,
            n=context.n_related_posts
        )
        if len(related_posts) < context.n_related_posts:
            related_posts2 = get_n_most_recent_posts_in_same_cluster(
                post_id=post.post_id,
                cluster_cardinality=5,
                n=context.n_related_posts*2
            )
            postids = [p.post_id for p in related_posts]
            related_posts = related_posts + [p for p in related_posts2 if p.post_id not in postids]
            related_posts = related_posts[:context.n_related_posts]

        # Always use synthesizer approach (removed bypass_synthesizer)
        synthesized_info = synthesize_context(
            new_post=post,
            previous_posts=related_posts,
            metric_name="Reasoning Quality",
            metric_evaluation_prompt=REASONING_QUALITY_EVALUATION_CRITERIA,
            model=context.model,
            synthesizer_focus_area=SYNTHESIZER_FOCUS_AREA
        )

        prompt = PROMPT_REASONING_QUALITY_V3.format(
            evaluation_criteria=REASONING_QUALITY_EVALUATION_CRITERIA,
            title=post.title,
            post_text=post_text,
            synthesized_info=synthesized_info
        )

        response = client.chat.completions.create(
            model=context.model,
            messages=[{"role": "user", "content": prompt}]
        )

        raw_content = response.choices[0].message.content
        result = parse_json_with_repair(raw_content)

        return cls(
            post_id=post.post_id,
            reasoning_quality_score=result["reasoning_quality_score"],
            thesis=result["thesis"],
            logical_arguments=result["logical_arguments"],
            explanation=result["explanation"]
        )


# Extract the evaluation criteria for the synthesizer
REASONING_QUALITY_EVALUATION_CRITERIA = """Evaluate the quality of reasoning in this EA Forum post.

First identify the thesis of the post. Then identify the logical arguments made to support the thesis. Finally assess how effective the arguments are at supporting the thesis.

Step 1: What is the main thesis or central argument of this post? (Be specific and concise)

Step 2: What logical arguments does the post make? Consider:
- Deductive reasoning and logical inferences
- Analogies and comparative reasoning
- Causal arguments and mechanism explanations
- Theoretical frameworks and conceptual analysis

Step 3: How effectively does the reasoning support the thesis? Once you've identified the thesis and arguments, consider:
- How logically sound and well-structured are the arguments?
- Are the logical connections between premises and conclusions valid?
- Does the reasoning flow coherently from point to point?
- Are there any reasoning errors, gaps, or unsupported dubious assumptions?
- Are there plausible counter-arguments, and are these adequately addressed?

Consider:
- Logical soundness: Are the inferences valid?
- Sufficiency: Is there enough reasoning to support the conclusion?
- Efficiency: Is the reasoning appropriately concise vs redundant?
- Structure: Does the logic flow coherently?
- Rigor: How analytically robust is the thinking?

Also consider any relevant context provided by the synthesis agent on related previous posts and how this context may bear sufficiency, efficiency, and structure of the reasoning in this post.

Rubric:
Grade on a 1-10 scale for OPTIMAL reasoning quality:
- 1-2: No coherent reasoning OR deeply flawed logical structure
- 3-4: Weak reasoning - significant logical gaps, fallacies, or unsound arguments
- 5-6: Moderate reasoning - some solid arguments but notable weaknesses or gaps
- 7-8: Strong reasoning - well-structured arguments that effectively support the thesis
- 9-10: Optimally reasoned - rigorous logic with just the right level of argumentation, neither under-argued nor redundantly over-argued
"""

SYNTHESIZER_FOCUS_AREA = """Look for:
- Whether previous posts provide evidence or reasoning that renders reasoning in this post more or less valid or more or less valuable
- Whether previous posts provide counter-arguments or objections and whether this post addresses them
"""


PROMPT_REASONING_QUALITY_V3 = """{evaluation_criteria}

Synthesis agent compiled some potentially useful context from recent, related posts:
```
{synthesized_info}
```

Post content to grade:
```
{title}
{post_text}
```

Respond with JSON:
```json{{
    "thesis": "<main thesis/argument of the post>",
    "logical_arguments": "<summary of key logical arguments made>",
    "explanation": "<explanation of how effectively the reasoning supports the thesis>"
    "reasoning_quality_score": <int 1-10>,
}}
```
"""


def compute_reasoning_quality_v3(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini",
    bypass_synthesizer: bool = False,
    n_related_posts: int = 5,
) -> ReasoningQualityV3:
    """Compute reasoning quality score for a post with synthesized information from related posts.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        bypass_synthesizer: If True, use raw related posts instead of synthesized context
        
    Returns:
        ReasoningQualityV3 metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    related_posts = get_n_most_recent_posts_in_same_cluster(
        post_id=post.post_id,
        cluster_cardinality=12,
        n=n_related_posts
    )
    if len(related_posts) < n_related_posts:
        related_posts2 = get_n_most_recent_posts_in_same_cluster(
            post_id=post.post_id,
            cluster_cardinality=5,
            n=n_related_posts*2
        )
        postids = [p.post_id for p in related_posts]
        related_posts = related_posts + [p for p in related_posts2 if p.post_id not in postids]
        related_posts = related_posts[:n_related_posts]

    # Use either synthesizer or raw related posts formatting
    if bypass_synthesizer:
        synthesized_info = format_raw_related_posts(related_posts)
    else:
        synthesized_info = synthesize_context(
            new_post=post,
            previous_posts=related_posts,
            metric_name="Reasoning Quality",
            metric_evaluation_prompt=REASONING_QUALITY_EVALUATION_CRITERIA,
            model=model,
            synthesizer_focus_area=SYNTHESIZER_FOCUS_AREA
        )

    prompt = PROMPT_REASONING_QUALITY_V3.format(
        evaluation_criteria=REASONING_QUALITY_EVALUATION_CRITERIA,
        title=post.title,
        post_text=post_text,
        synthesized_info=synthesized_info
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    result = parse_json_with_repair(raw_content)
    
    return ReasoningQualityV3(
        post_id=post.post_id,
        reasoning_quality_score=result["reasoning_quality_score"],
        thesis=result["thesis"],
        logical_arguments=result["logical_arguments"],
        explanation=result["explanation"]
    )