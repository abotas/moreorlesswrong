from typing import Literal
from pydantic import BaseModel
import json

from models import Post, Claim
from llm_client import client


class NoveltySupport(BaseModel):
    post_id: str
    claim_id: str
    novelty_support_ea: int  # 1-10 score for EA forum readership
    novelty_support_humanity: int  # 1-10 score for all humanity
    explanation: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "NoveltySupport"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["novelty_support_ea", "novelty_support_humanity"]


PROMPT_NOVELTY_SUPPORT = """Evaluate the novelty of the APPROACH, METHOD, TYPE OF EVIDENCE, or REASONING used to support this claim from an EA Forum post.

Focus on how novel the WAY the post supports the claim is, NOT how novel the claim itself is.

Consider the novelty of:
- The type of evidence provided
- The reasoning methodology used  
- The analytical approach taken
- The combination of sources/perspectives
- The argumentation structure

Score the novelty on a 1-10 scale for two audiences:
1. EA Forum readership (people familiar with effective altruism, rationality, longtermism)
2. General humanity (average educated person)

Scale:
- 1: Extremely common/standard approach that almost everyone uses
- 5: Moderately novel approach, some people have used this method
- 10: Highly novel/original approach that very few have tried

Respond with a JSON object:
{{
    "novelty_support_ea": <int 1-10>,
    "novelty_support_humanity": <int 1-10>,
    "explanation": "<brief explanation of your scoring>"
}}

The claim:
"{claim}"

The full post in which to evaluate the novelty of the support:
{post_title}

{post_text}
"""


def compute_novelty_support(
    claim: Claim,
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> NoveltySupport:
    """Compute novelty support scores for a claim's supporting approach/method.
    
    Args:
        claim: The claim to evaluate
        post: The post containing the claim
        model: The model to use for evaluation
        
    Returns:
        NoveltySupport metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    prompt = PROMPT_NOVELTY_SUPPORT.format(
        claim=claim.claim,
        post_title=post.title,
        post_text=post_text
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
    
    return NoveltySupport(
        post_id=post.post_id,
        claim_id=claim.claim_id,
        novelty_support_ea=result["novelty_support_ea"],
        novelty_support_humanity=result["novelty_support_humanity"],
        explanation=result["explanation"]
    )