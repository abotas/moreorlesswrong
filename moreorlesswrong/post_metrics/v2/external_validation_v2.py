from typing import Literal
from pydantic import BaseModel
import json

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair


class ExternalValidationV2(BaseModel):
    post_id: str
    external_validation_score: int  # 1-10 overall external validation score
    explanation: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "ExternalValidationV2"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["external_validation_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "external_validation_score": "External Validation Score"
        }


PROMPT_EXTERNAL_VALIDATION = """
You are an expert validator of claims. Rate the overall external validation of this post using trustworthy sources found online that corroborate or contradict the claims.

First identify the major claims in the post that should be to some extent verifiable via external sources. Then assess how well external sources validate those claims.
Bear in mind some claims are inherently not verifiable and those claims should be excluded from this assessment. 

Consider:
- Do authoritative sources support or contradict the key verifiable assertions?
- Are statistics, data points, and research findings accurately represented?

Post content to grade:
{title}

{post_text}

Rubric:
Grade on a 1-10 scale for external validation:
- 1-2: Strong evidence contradicts the central claims
- 3-4: Mostly inaccurate, significant evidence against key claims
- 5-6: Mixed/uncertain, conflicting evidence around key claims. OR there are no key claims that should be subject to external verification
- 7-8: Solid evidence validates the major claims requiring external verification
- 9-10: Optimally externally validated, the major claims that are verifiable are fully validated. The semi-verifiable claims fit plausibly are plausible given evidence available in the world.

Low scores indicate contradiction with authoritative and generally trusted sources. Moderate scores indicate conflicting evidence or insufficient verification. High scores indicate well-supported evidence that validates the major claims requiring external verification.

Respond with JSON:
{{
    "external_validation_score": <int 1-10 overall validation>,
    "explanation": "<brief assessment including specific sources consulted>"
}}
"""


def compute_external_validation_v2(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> ExternalValidationV2:
    """Compute external validation scores for a post using web search.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        ExternalValidationV2 metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    prompt = PROMPT_EXTERNAL_VALIDATION.format(
        title=post.title,
        post_text=post_text
    )
    
    # Use GPT-5 with web search capability using responses API
    response = client.responses.create(
        model=model,
        tools=[{"type": "web_search_preview"}],
        input=prompt
    )
    
    raw_content = response.output_text
    result = parse_json_with_repair(raw_content)
    
    return ExternalValidationV2(
        post_id=post.post_id,
        external_validation_score=result["external_validation_score"],
        explanation=result["explanation"]
    )