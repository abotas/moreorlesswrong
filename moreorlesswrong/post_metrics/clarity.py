from typing import Literal
from pydantic import BaseModel
import json

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair


class PostClarity(BaseModel):
    post_id: str
    clarity_score: int  # 1-10 overall clarity score
    explanation: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "PostClarity"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["clarity_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "clarity_score": "Clarity Score"
        }


PROMPT_POST_CLARITY = """Evaluate the clarity of this EA Forum post.

Consider:
1. Comprehensibility: How easy is it to understand the post?
2. Argument Clarity: Does it make its point in a clear and compelling way?
3. Conciseness: Is it appropriately concise given the information it's presenting?

Post title: {title}

Post content:
{post_text}

Rate on a 1-10 scale:
- 1-2: Very unclear, confusing, rambling or unnecessarily verbose
- 3-4: Somewhat unclear, requires significant effort to understand
- 5-6: Moderately clear, understandable with some effort
- 7-8: Clear, easy to follow, well-structured
- 9-10: Exceptionally clear, compelling, and perfectly concise

Respond with JSON:
{{
    "clarity_score": <int 1-10 overall clarity>,
    "explanation": "<brief explanation of clarity strengths and weaknesses>"
}}
"""


def compute_post_clarity(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> PostClarity:
    """Compute clarity scores for a post.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        PostClarity metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    prompt = PROMPT_POST_CLARITY.format(
        title=post.title,
        post_text=post_text
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    result = parse_json_with_repair(raw_content)
    
    return PostClarity(
        post_id=post.post_id,
        clarity_score=result["clarity_score"],
        explanation=result["explanation"]
    )