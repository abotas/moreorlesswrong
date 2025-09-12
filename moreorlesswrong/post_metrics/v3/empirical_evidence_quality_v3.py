from typing import Literal
from pydantic import BaseModel

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair
from db import get_n_most_recent_posts_in_same_cluster
from synthesizer import synthesize_context


class EmpiricalEvidenceQualityV3(BaseModel):
    post_id: str
    empirical_evidence_quality_score: int  # 1-10 empirical evidence quality score
    thesis: str  # The identified main thesis
    empirical_claims: str  # Key empirical claims identified
    explanation: str  # How well claims support thesis
    
    @classmethod
    def metric_name(cls) -> str:
        return "EmpiricalEvidenceQualityV3"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["empirical_evidence_quality_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "empirical_evidence_quality_score": "Empirical Evidence Quality Score"
        }


# Extract the evaluation criteria for the synthesizer
EMPIRICAL_EVIDENCE_QUALITY_EVALUATION_CRITERIA = """Evaluate the EMPIRICAL EVIDENCE QUALITY of this EA Forum post.

This is a three-step evaluation:
1. Identify the main thesis/argument of the post
2. Identify the empirical claims made to support that thesis
3. Evaluate how effectively the empirical evidence supports the thesis

Step 1: What is the main thesis or central argument of this post? (Be specific and concise)

Step 2: What empirical claims does the post make? Consider:
- Statistical data and quantitative claims
- Research findings and study results
- Historical examples and case studies
- Observable facts and measurements
- Causal claims about the world

Step 3: How effectively do these empirical claims support the thesis?

Rubric:
Grade on a 1-10 scale for empirical evidence quality:
- 1-2: No empirical evidence OR evidence contradicts thesis
- 3-4: Weak empirical support - few claims, poorly connected to thesis, or cherry-picked
- 5-6: Moderate support - some relevant evidence but gaps in the argument
- 7-8: Strong support - multiple relevant empirical claims that build a compelling case
- 9-10: Exceptional support - comprehensive, rigorous empirical evidence that decisively supports thesis

Consider:
- Relevance: Do the empirical claims actually address the thesis?
- Sufficiency: Is there enough evidence to support the conclusion?
- Representativeness: Is the evidence cherry-picked or comprehensive?
- Logical connection: Does the evidence logically lead to the thesis?
- Strength of inference: How strong is the link between evidence and conclusion?"""


PROMPT_EMPIRICAL_EVIDENCE_QUALITY_V3 = """{evaluation_criteria}

Synthesis agent compiled some potentially useful context from recent, related posts:
{synthesized_info}

Post content to grade:
```
{title}
{post_text}
```

Respond with JSON:
{{
    "thesis": "<main thesis/argument of the post>",
    "empirical_claims": "<summary of key empirical claims made>",
    "explanation": "<explanation of how well the empirical evidence supports the thesis>"
    "empirical_evidence_quality_score": <int 1-10>,
}}
"""


def compute_empirical_evidence_quality_v3(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini",
) -> EmpiricalEvidenceQualityV3:
    """Compute empirical evidence quality score for a post with synthesized information from related posts.
    
    This metric evaluates:
    1. The main thesis of the post
    2. The empirical claims made
    3. How well those claims support the thesis
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        EmpiricalEvidenceQualityV3 metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    # Get 5 most recent posts from the same cluster-5
    related_posts = get_n_most_recent_posts_in_same_cluster(
        post_id=post.post_id,
        cluster_cardinality=5,
        n=5
    )

    synthesized_info = synthesize_context(
        new_post=post,
        previous_posts=related_posts,
        metric_name="Empirical Evidence Quality",
        metric_evaluation_prompt=EMPIRICAL_EVIDENCE_QUALITY_EVALUATION_CRITERIA,
        model=model
    )

    prompt = PROMPT_EMPIRICAL_EVIDENCE_QUALITY_V3.format(
        evaluation_criteria=EMPIRICAL_EVIDENCE_QUALITY_EVALUATION_CRITERIA,
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
    
    return EmpiricalEvidenceQualityV3(
        post_id=post.post_id,
        empirical_evidence_quality_score=result["empirical_evidence_quality_score"],
        thesis=result["thesis"],
        empirical_claims=result["empirical_claims"],
        explanation=result["explanation"]
    )