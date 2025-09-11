from typing import Literal
from pydantic import BaseModel
import json

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair


class PrecisionV2(BaseModel):
    post_id: str
    precision_score: int  # 1-10 overall precision score
    explanation: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "PrecisionV2"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["precision_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "precision_score": "Precision Score"
        }


PROMPT_PRECISION = """Evaluate the PRECISION of this post for EA forum readership.

Precision is a measure of how informative the claims of a post are, in the sense of Shannon information. The higher precision, the more possible ways the world could be that the post rules out.

Being appropriately Precise requires saying the most informative thing you can that brings marginal value, without sacrificing truthfulness.

Although more precise information typically increases value, sometimes marginal information only makes a difference to inquiries we don't care about. For example, if I am choosing a college, I don't need to be informed about the building materials of the registrar's office.

Consider the following post. Go section by section through the post and asses how appropriate the level of precision is given its EA forum readership.

Post content to grade:
{title}
{post_text}

Rubric:
Grade on a 1-10 scale for OPTIMAL precision to *EA forum readership*:
- 1-2: Very poor precision balance - either too vague (platitudes) OR too detailed (superfluous minutiae)
- 3-4: Poor precision calibration - either underspecified OR cluttered with irrelevant detail
- 5-6: Moderate precision - some sections appropriately precise, others miss the mark
- 7-8: Well-calibrated precision - appropriately informative for the context and audience
- 9-10: Optimally precise - perfect balance of informativeness and relevance, neither too vague nor unnecessarily detailed

Respond with JSON:
{{
    "precision_score": <int 1-10 overall precision>,
    "explanation": "<brief explanation of precision assessment>"
}}
"""


def compute_precision_v2(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> PrecisionV2:
    """Compute precision scores for a post using epistemic virtue principles.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        PrecisionV2 metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    prompt = PROMPT_PRECISION.format(
        title=post.title,
        post_text=post_text
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    result = parse_json_with_repair(raw_content)
    
    return PrecisionV2(
        post_id=post.post_id,
        precision_score=result["precision_score"],
        explanation=result["explanation"]
    )