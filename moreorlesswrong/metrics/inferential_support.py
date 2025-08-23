from typing import Literal
from pydantic import BaseModel
import json

from models import Post, Claim
from llm_client import client


class InferentialSupport(BaseModel):
    post_id: str
    claim_id: str
    inferential_support: int  # 1-10 score for quality of reasoning and evidence
    explanation: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "InferentialSupport"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["inferential_support"]


PROMPT_INFERENTIAL_SUPPORT = """Evaluate how well the following claim is supported by reasoning and evidence in the EA Forum post.

Consider:
- Quality of logical arguments presented *that support the claim*
- Empirical evidence provided (data, studies, examples) *that support the claim*
- Coherence of reasoning chain *that supports the claim*
- Acknowledgment of counterarguments or limitations
- Depth of analysis

Score on a 1-10 scale:
- 1: No support - bare assertion with no reasoning or evidence
- 3: Minimal support - some reasoning but weak or flawed
- 5: Moderate support - decent reasoning or some evidence
- 7: Good support - solid reasoning and/or strong evidence
- 10: Excellent support - rigorous reasoning with compelling evidence

Respond with a JSON object:
{{
    "inferential_support": <int 1-10>,
    "explanation": "<brief explanation of what support exists or is lacking>"
}}

Context from the post:
Title: {title}
Author: {author}

The claim to evaluate:
"{claim}"

Full post to evaluate inferential support for the claim:
{post_text}
"""


def compute_inferential_support(
    claim: Claim,
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> InferentialSupport:
    """Compute inferential support score for a claim.
    
    Args:
        claim: The claim to evaluate
        post: The post containing the claim
        model: The model to use for evaluation
        
    Returns:
        InferentialSupport metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    prompt = PROMPT_INFERENTIAL_SUPPORT.format(
        title=post.title,
        author=post.author_display_name or "Unknown",
        claim=claim.claim,
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
    
    return InferentialSupport(
        post_id=post.post_id,
        claim_id=claim.claim_id,
        inferential_support=result["inferential_support"],
        explanation=result["explanation"]
    )