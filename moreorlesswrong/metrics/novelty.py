from typing import Literal
from pydantic import BaseModel
import json

from models import Post, Claim
from llm_client import client


class Novelty(BaseModel):
    post_id: str
    claim_id: str
    novelty_ea: int  # 1-10 score for EA forum readership
    novelty_humanity: int  # 1-10 score for all humanity
    explanation: str


PROMPT_NOVELTY = """Evaluate the novelty of the following claim from an EA Forum post.

Novelty means how NEW or ORIGINAL this idea/claim is - something is novel if people are unlikely to have already considered it.

Score the novelty on a 1-10 scale for two audiences:
1. EA Forum readership (people familiar with effective altruism, rationality, longtermism)
2. General humanity (average educated person)

Scale:
- 1: Extremely common/obvious idea that almost everyone has considered
- 5: Moderately novel, some people have thought about this
- 10: Highly novel/original idea that very few have considered

Respond with a JSON object:
{{
    "novelty_ea": <int 1-10>,
    "novelty_humanity": <int 1-10>,
    "explanation": "<brief explanation of your scoring>"
}}

Context from the post:
Title: {title}
Author: {author}

The claim to evaluate:
"{claim}"

Full post for context:
{post_text}
"""


def compute_novelty(
    claim: Claim,
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> Novelty:
    """Compute novelty scores for a claim.
    
    Args:
        claim: The claim to evaluate
        post: The post containing the claim
        model: The model to use for evaluation
        
    Returns:
        Novelty metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    prompt = PROMPT_NOVELTY.format(
        title=post.title,
        author=post.author_display_name or "Unknown",
        claim=claim.claim,
        post_text=post_text[:5000]  # Limit context length
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    
    # Strip potential markdown formatting
    if raw_content.startswith("```"):
        raw_content = raw_content.split("```")[1]
        if raw_content.startswith("json"):
            raw_content = raw_content[4:]
        raw_content = raw_content.strip()
    
    result = json.loads(raw_content)
    
    return Novelty(
        post_id=post.post_id,
        claim_id=claim.claim_id,
        novelty_ea=result["novelty_ea"],
        novelty_humanity=result["novelty_humanity"],
        explanation=result["explanation"]
    )