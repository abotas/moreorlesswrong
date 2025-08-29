from typing import Literal
from pydantic import BaseModel

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair


class PostNovelty(BaseModel):
    post_id: str
    novelty_ea: int  # 1-10 score for EA forum readership
    novelty_humanity: int  # 1-10 score for all humanity
    explanation: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "PostNovelty"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["novelty_ea", "novelty_humanity"]


PROMPT_POST_NOVELTY = """Evaluate the overall novelty of the following EA Forum post.

Novelty means how NEW or ORIGINAL the main ideas/claims in this post are - something is novel if people are unlikely to have already considered it.

Score the novelty on a 1-10 scale for two audiences:
1. EA Forum readership (people familiar with effective altruism, rationality, longtermism)
2. General humanity (average educated person)

Scale:
- 1: Extremely common/obvious ideas that almost everyone has considered
- 5: Moderately novel, some people have thought about this
- 10: Highly novel/original ideas that very few have considered

Post title: {title}

Post content:
{post_text}

Respond with a JSON object:
{{
    "novelty_ea": <int 1-10>,
    "novelty_humanity": <int 1-10>,
    "explanation": "<brief explanation of your scoring focusing on the most novel aspects>"
}}
"""


def compute_post_novelty(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> PostNovelty:
    """Compute novelty scores for an entire post.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        PostNovelty metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    prompt = PROMPT_POST_NOVELTY.format(
        title=post.title,
        post_text=post_text
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    result = parse_json_with_repair(raw_content)
    
    return PostNovelty(
        post_id=post.post_id,
        novelty_ea=result["novelty_ea"],
        novelty_humanity=result["novelty_humanity"],
        explanation=result["explanation"]
    )