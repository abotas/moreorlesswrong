from typing import Literal
from pydantic import BaseModel
import json

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair


class PostNoveltySupport(BaseModel):
    post_id: str
    novelty_support: int  # 1-10 score for novelty of support methodology
    explanation: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "PostNoveltySupport"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["novelty_support"]


PROMPT_POST_NOVELTY_SUPPORT = """Evaluate the novelty of the support methodology for this EA Forum post.

Focus on how novel the WAY the post makes its argument is, NOT how novel the thesis itself is.

Consider the novelty of:
- The type of evidence provided
- The reasoning methodology used
- The analytical approach taken
- The combination of sources/perspectives used
- The argumentation structure

Post title: {title}

Post content:
{post_text}

Score the novelty on a 1-10 scale:
- 1: Extremely common/standard approach that almost everyone uses
- 5: Moderately novel approach, some people have used this method
- 10: Highly novel/original approach that very few have tried

Respond with a JSON object:
{{
    "novelty_support": <int 1-10>,
    "explanation": "<brief explanation of what makes the support methodology novel or conventional>"
}}
"""


def compute_post_novelty_support(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> PostNoveltySupport:
    """Compute novelty support scores for a post's supporting approach/method.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        PostNoveltySupport metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    prompt = PROMPT_POST_NOVELTY_SUPPORT.format(
        title=post.title,
        post_text=post_text
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    result = parse_json_with_repair(raw_content)
    
    return PostNoveltySupport(
        post_id=post.post_id,
        novelty_support=result["novelty_support"],
        explanation=result["explanation"]
    )