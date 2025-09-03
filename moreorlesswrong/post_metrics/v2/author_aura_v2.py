from typing import Literal
from pydantic import BaseModel
import json

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair


class AuthorAuraV2(BaseModel):
    post_id: str
    ea_fame_score: int  # 1-10 EA fame score
    explanation: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "AuthorAuraV2"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["ea_fame_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "ea_fame_score": "EA Fame Score"
        }


PROMPT_AUTHOR_AURA = """Evaluate the EA FAME of the author of this EA Forum post.

EA Fame refers to how well-known and influential the author is within the Effective Altruism community.

Consider:
- Is this a recognized thought leader in EA?
- Have they made significant contributions to EA discourse?
- Are they frequently cited or referenced in EA discussions?
- Do they hold influential positions in EA organizations?

Common indicators of high EA fame:
- Founders/leaders of major EA organizations (Open Philanthropy, CEA, etc.)
- Authors of highly influential EA posts or books
- Frequent speakers at EA conferences
- Recipients of major EA grants or awards


Author: {author_name}

Post content (for context):
{title}

{post_text}

Rubric:
Grade on a 1-10 scale for EA fame:
- 1-2: Unknown in EA community
- 3-4: Minor presence, occasional contributor
- 5-6: Moderately known, regular contributor
- 7-8: Well-known, influential voice
- 9-10: EA thought leader, highly influential

Respond with JSON:
{{
    "ea_fame_score": <int 1-10 EA fame>,
    "explanation": "<brief explanation of EA fame assessment>"
}}
"""


def compute_author_aura_v2(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> AuthorAuraV2:
    """Compute author EA fame score for a post.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        AuthorAuraV2 metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    prompt = PROMPT_AUTHOR_AURA.format(
        title=post.title,
        author_name=post.author_display_name or "Unknown",
        post_text=post_text
    )
    # No websearch bc of data leakage
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    result = parse_json_with_repair(raw_content)
    
    return AuthorAuraV2(
        post_id=post.post_id,
        ea_fame_score=result["ea_fame_score"],
        explanation=result["explanation"]
    )