from typing import Literal
from pydantic import BaseModel
import json

from models import Post, Claim
from llm_client import client


class ExternalValidation(BaseModel):
    post_id: str
    claim_id: str
    external_validation: int  # 1-10 score for external source validation
    explanation: str
    sources: list[str]
    
    @classmethod
    def metric_name(cls) -> str:
        return "ExternalValidation"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["external_validation"]


PROMPT_EXTERNAL_VALIDATION = """You are an expert fact-checker evaluating claims from an EA Forum post. 
Rate the veracity of this claim on a 1-10 scale using your knowledge and trustworthy sources found online.
ALWAYS cite specific sources for your assessment.

Consider:
- Scientific studies and peer-reviewed research
- Official statistics and government data
- Reports from credible organizations
- Expert consensus in relevant fields
- Historical facts and documented events

Scale:
- 1: Completely false - strong evidence contradicts this claim
- 3: Mostly false - significant evidence against this claim or part of this claim.
- 5: Mixed/uncertain - conflicting evidence or insufficient information
- 7: Mostly true - good evidence supports this claim
- 10: Completely true - overwhelming evidence supports this claim

Respond with valid JSON matching this schema:
{{"external_validation": <int 1-10>, "explanation": "<concise rationale with key evidence>", "sources": ["source1", "source2", ...]}}

Include specific sources in the sources array (e.g., "World Bank Development Report 2023", "Nature study by Smith et al., 2024", "WHO Global Health Statistics").

Context:
Post: "{title}" by {author}

Claim to evaluate:
"{claim}"
"""


def compute_external_validation(
    claim: Claim,
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> ExternalValidation:
    """Compute external validation score for a claim using web search.
    
    Args:
        claim: The claim to evaluate
        post: The post containing the claim
        model: The model to use for evaluation
        
    Returns:
        ExternalValidation metric object
    """
    prompt = PROMPT_EXTERNAL_VALIDATION.format(
        title=post.title,
        author=post.author_display_name or "Unknown",
        claim=claim.claim
    )
    
    # Use GPT-5 with web search capability using responses API
    response = client.responses.create(
        model=model,
        tools=[{"type": "web_search_preview"}],
        input=prompt
    )
    
    raw_content = response.output_text
    
    # Strip potential markdown formatting
    if raw_content.startswith("```"):
        raw_content = raw_content.split("```")[1]
        if raw_content.startswith("json"):
            raw_content = raw_content[4:]
        raw_content = raw_content.strip()
    
    result = json.loads(raw_content)
    
    return ExternalValidation(
        post_id=post.post_id,
        claim_id=claim.claim_id,
        external_validation=result["external_validation"],
        explanation=result["explanation"],
        sources=result.get("sources", [])  # Use get() in case sources aren't returned
    )