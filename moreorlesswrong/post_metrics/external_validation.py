from typing import Literal
from pydantic import BaseModel
import json

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair


class PostExternalValidation(BaseModel):
    post_id: str
    emperical_claim_validation_score: int  # 1-10 overall external validation score
    validation_notes: str
    sources: list[str]
    
    @classmethod
    def metric_name(cls) -> str:
        return "PostExternalValidation"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["emperical_claim_validation_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "emperical_claim_validation_score": "External Validation Score"
        }


PROMPT_POST_EXTERNAL = """You are an expert evaluator of emperical claims. You're given an EA Forum post and asked to evaluate 
the the accuracy of the empirical claims made in the post.
Rate the overall validation of this post on a 1-10 scale using your knowledge and trustworthy sources found online.
ALWAYS cite specific sources for your assessment.

Consider:
- Can the major claims made in the post be verified through reliable external sources in websearch?

Scale:
- 1: Many factual errors - strong evidence contradicts major claims.
- 3: Mostly inaccurate - significant evidence against key claims
- 5: Mixed/uncertain - conflicting evidence or insufficient verification
- 7: Well-supported - good evidence validates most major claims
- 10: Exceptionally validated - overwhelming evidence supports all key claims

Post title: {title}
Author: {author}

Post content:
{post_text}

Respond with valid JSON:
{{
    "emperical_claim_validation_score": <int 1-10>,
    "validation_notes": "<concise assessment of validation strengths and weaknesses>",
    "sources": ["source1", "source2", ...]
}}

Include specific sources you consulted in the sources array (e.g., "WHO Global Health Observatory 2024", "Nature study by Johnson et al., 2023", "World Bank data").
"""


def compute_post_external_validation(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> PostExternalValidation:
    """Compute external validation scores for a post.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        PostExternalValidation metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    prompt = PROMPT_POST_EXTERNAL.format(
        title=post.title,
        author=post.author_display_name or "Unknown",
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
    
    return PostExternalValidation(
        post_id=post.post_id,
        emperical_claim_validation_score=result["emperical_claim_validation_score"],
        validation_notes=result["validation_notes"],
        sources=result.get("sources", [])  # Use get() in case sources aren't returned
    )