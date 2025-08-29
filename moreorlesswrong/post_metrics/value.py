from typing import Literal
from pydantic import BaseModel

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair


class PostValue(BaseModel):
    post_id: str
    value_ea: int  # 1-10 importance score for EA community
    value_humanity: int  # 1-10 importance score for humanity
    explanation: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "PostValue"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["value_ea", "value_humanity"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "value_ea": "Value to EA Community",
            "value_humanity": "Value to Humanity"
        }


PROMPT_POST_VALUE = """Evaluate how important/valuable this EA Forum post is overall.

Consider how load-bearing, useful, and critical the main thesis and arguments are - if the post's conclusions are true/false, how much would it matter?

Rate the importance on a 1-10 scale for two audiences:
1. EA/rationalist community (effective altruism, longtermism, AI safety, global health, etc.)
2. General humanity

Scale:
- 1: Trivial/irrelevant - post has no meaningful impact
- 3: Minor importance - might affect some decisions or understanding 
- 5: Moderate importance - would influence several important decisions or beliefs
- 7: High importance - load-bearing for major decisions or worldview
- 10: Critical importance - foundational ideas that many other important conclusions depend on

Consider:
- Whether this post's thesis is foundational to other important conclusions
- The scale and scope of potential impact
- The practical implications if the post is correct

Post title: {title}

Post content:
{post_text}

Respond with a JSON object:
{{
    "value_ea": <int 1-10>,
    "value_humanity": <int 1-10>,
    "explanation": "<brief explanation of the importance/impact of this post>"
}}
"""


def compute_post_value(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> PostValue:
    """Compute value/importance scores for an entire post.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        PostValue metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    prompt = PROMPT_POST_VALUE.format(
        title=post.title,
        post_text=post_text
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    result = parse_json_with_repair(raw_content)
    
    return PostValue(
        post_id=post.post_id,
        value_ea=result["value_ea"],
        value_humanity=result["value_humanity"],
        explanation=result["explanation"]
    )