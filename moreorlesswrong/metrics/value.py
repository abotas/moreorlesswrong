from typing import Literal
from pydantic import BaseModel
import json

from models import Post, Claim
from llm_client import client


class Value(BaseModel):
    post_id: str
    claim_id: str
    value_ea: int  # 1-10 importance score for EA community
    value_humanity: int  # 1-10 importance score for humanity
    explanation: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "Value"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["value_ea", "value_humanity"]


PROMPT_VALUE = """Evaluate how important/valuable this claim is from an EA Forum post.

Consider how load-bearing, useful, and critical this claim is - if this claim is true/false, how much would it matter?

Rate the importance on a 1-10 scale for two audiences:
1. EA/rationalist community (effective altruism, longtermism, AI safety, global health, etc.)
2. General humanity

Scale:
- 1: Trivial/irrelevant - claim has no meaningful impact
- 3: Minor importance - might affect some decisions or understanding 
- 5: Moderate importance - would influence several important decisions or beliefs
- 7: High importance - load-bearing for major decisions or worldview
- 10: Critical importance - foundational claim that many other important conclusions depend on

Consider:
- How many people/decisions would be affected if this claim is true/false
- Whether this claim is foundational to other important conclusions
- The scale and scope of potential impact
- Whether this fills an important knowledge gap

Respond with a JSON object:
{{
    "value_ea": <int 1-10>,
    "value_humanity": <int 1-10>,
    "explanation": "<brief explanation of the importance/impact of this claim>"
}}

The claim to evaluate:
"{claim}"
"""


def compute_value(
    claim: Claim,
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> Value:
    """Compute value/importance scores for a claim.
    
    Args:
        claim: The claim to evaluate
        post: The post containing the claim
        model: The model to use for evaluation
        
    Returns:
        Value metric object
    """
    prompt = PROMPT_VALUE.format(claim=claim.claim)
    
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
    
    return Value(
        post_id=post.post_id,
        claim_id=claim.claim_id,
        value_ea=result["value_ea"],
        value_humanity=result["value_humanity"],
        explanation=result["explanation"]
    )