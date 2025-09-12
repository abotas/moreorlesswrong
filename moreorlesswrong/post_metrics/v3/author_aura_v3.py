from typing import Literal
from pydantic import BaseModel
from datetime import datetime, timezone
import json

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair
from db import get_posts_by_author_in_date_range


class AuthorAuraV3(BaseModel):
    post_id: str
    ea_fame_score: int  # 1-10 EA fame score
    explanation: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "AuthorAuraV3"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["ea_fame_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "ea_fame_score": "EA Fame Score"
        }


PROMPT_AUTHOR_AURA_V3 = """Evaluate the EA FAME of the author of this EA Forum post.

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

Author's posting history:
- Posts from 2024-01-01 to current post date: {num_posts}
- Average karma (base_score) of those posts: {avg_karma:.1f}

Post content (for context):
```
{title}
{post_text}
```

Rubric:
Grade on a 1-10 scale for EA fame:
- 1-2: Unknown in EA community
- 3-4: Minor presence, occasional contributor
- 5-6: Moderately known, regular contributor
- 7-8: Well-known, influential voice
- 9-10: EA thought leader, highly influential

Note: The author's posting history (number of posts and average karma) provides additional context about their engagement and reception in the EA community.

Respond with JSON:
{{
    "explanation": "<brief explanation of EA fame assessment>"
    "ea_fame_score": <int 1-10 EA fame>,
}}
"""


def compute_author_aura_v3(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> AuthorAuraV3:
    """Compute author EA fame score for a post, including author's posting history.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        AuthorAuraV3 metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_date = post.posted_at
    
    author_posts = get_posts_by_author_in_date_range(
        author_display_name=post.author_display_name or "Unknown",
        start_date=start_date,
        end_date=end_date
    )
    
    # Calculate statistics
    num_posts = len(author_posts)
    avg_karma = 0.0
    if author_posts:
        karma_values = [p.base_score for p in author_posts if p.base_score is not None]
        if karma_values:
            avg_karma = sum(karma_values) / len(karma_values)
    
    prompt = PROMPT_AUTHOR_AURA_V3.format(
        title=post.title,
        author_name=post.author_display_name or "Unknown",
        num_posts=num_posts,
        avg_karma=avg_karma,
        post_text=post_text
    )
    
    # No websearch bc of data leakage
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    result = parse_json_with_repair(raw_content)
    
    return AuthorAuraV3(
        post_id=post.post_id,
        ea_fame_score=result["ea_fame_score"],
        explanation=result["explanation"]
    )