from typing import Literal
from pydantic import BaseModel

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair


class TitleClickabilityV2(BaseModel):
    post_id: str
    title_clickability_score: int  # 1-10 title clickability score
    explanation: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "TitleClickabilityV2"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["title_clickability_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "title_clickability_score": "Title Clickability Score"
        }


PROMPT_TITLE_CLICKABILITY = """Evaluate the CLICKABILITY of this EA Forum post title.

Title clickability refers to how compelling the title is to make someone want to read the post, while maintaining EA standards of intellectual honesty (not pure clickbait).

Rubric:
Grade on a 1-10 scale for title clickability.

High clickability titles (7-10):
- Create curiosity gaps ("We're spending $10M wrong")
- Make bold, specific claims ("AI safety funding should increase 10x by 2025")
- Promise valuable insights ("What I learned from analyzing 1000 EA careers")
- Challenge assumptions ("Why EA's biggest success might be its biggest failure")
- Quantify impact ("This simple change could save 100,000 lives")
- Reveal insider knowledge ("What OpenAI doesn't want EAs to know")

Medium clickability titles (4-6):
- Clear but conventional ("An analysis of charity effectiveness")
- Informative but dry ("Quarterly update on biosecurity research")
- Good topic but bland framing ("Thoughts on longtermism")
- Technical but relevant ("New method for cause prioritization")

Low clickability titles (1-3):
- Generic or vague ("Some considerations")
- Overly academic ("A preliminary investigation into...")
- No clear value proposition ("Random musings on EA")
- Pure jargon without context ("BOTEC of QALY/DALY ratios")

Post title: {title}

Consider:
- Does it create a curiosity gap?
- Is there a clear value proposition?
- Does it make specific, intriguing claims?
- Does it promise to resolve uncertainty?
- Would this stand out in a list of posts?

IMPORTANT: Good clickability in EA context means compelling but honest, not manipulative clickbait.

Respond with JSON:
{{
    "title_clickability_score": <int 1-10>,
    "explanation": "<brief explanation of what makes this title compelling or not>"
}}
"""


def compute_title_clickability_v2(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> TitleClickabilityV2:
    """Compute title clickability score for a post.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        TitleClickabilityV2 metric object
    """
    prompt = PROMPT_TITLE_CLICKABILITY.format(
        title=post.title
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    result = parse_json_with_repair(raw_content)
    
    return TitleClickabilityV2(
        post_id=post.post_id,
        title_clickability_score=result["title_clickability_score"],
        explanation=result["explanation"]
    )