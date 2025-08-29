from typing import Literal
from pydantic import BaseModel
import json

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair


class PostInferentialSupport(BaseModel):
    post_id: str
    reasoning_quality: int  # 1-10 for quality of reasoning
    evidence_quality: int  # 1-10 for quality of evidence
    overall_support: int  # 1-10 overall support quality
    explanation: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "PostInferentialSupport"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["reasoning_quality", "evidence_quality", "overall_support"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "reasoning_quality": "Reasoning Quality",
            "evidence_quality": "Evidence Quality", 
            "overall_support": "Overall Support"
        }


PROMPT_POST_INFERENTIAL = """Evaluate the quality of reasoning and evidence supporting the main thesis of this EA Forum post.

Consider:
1. Reasoning Quality: How logically sound and well-structured are the arguments?
2. Evidence Quality: How strong, relevant, and sufficient is the empirical evidence provided?
3. Overall Support: How well-supported is the main thesis overall?

Post title: {title}

Post content:
{post_text}

Rate on a 1-10 scale:
- 1: No support or deeply flawed reasoning/evidence
- 3: Weak support with significant gaps or errors
- 5: Moderate support with some solid points but notable weaknesses
- 7: Strong support with good reasoning and evidence
- 10: Exceptional support with rigorous reasoning and compelling evidence

Respond with JSON:
{{
    "reasoning_quality": <int 1-10>,
    "evidence_quality": <int 1-10>,
    "overall_support": <int 1-10>,
    "explanation": "<brief explanation of strengths and weaknesses in the support>"
}}
"""


def compute_post_inferential_support(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> PostInferentialSupport:
    """Compute inferential support scores for a post.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        PostInferentialSupport metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    prompt = PROMPT_POST_INFERENTIAL.format(
        title=post.title,
        post_text=post_text
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    result = parse_json_with_repair(raw_content)
    
    return PostInferentialSupport(
        post_id=post.post_id,
        reasoning_quality=result["reasoning_quality"],
        evidence_quality=result["evidence_quality"],
        overall_support=result["overall_support"],
        explanation=result["explanation"]
    )