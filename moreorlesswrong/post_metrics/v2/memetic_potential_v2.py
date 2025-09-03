from typing import Literal
from pydantic import BaseModel

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair


class MemeticPotentialV2(BaseModel):
    post_id: str
    memetic_potential_score: int  # 1-10 memetic potential score
    explanation: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "MemeticPotentialV2"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["memetic_potential_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "memetic_potential_score": "Memetic Potential Score"
        }


PROMPT_MEMETIC_POTENTIAL = """Evaluate the MEMETIC POTENTIAL of this EA Forum post.

Memetic potential refers to how likely this post is to:
- Create new terminology that others will adopt
- Introduce frameworks or schemas that become reference points
- Generate concepts that spread through the EA community
- Produce ideas that get cited and built upon

Rubric:
Grade on a 1-10 scale for memetic potential.

High memetic potential (7-10):
- Coins compelling new terms or acronyms
- Presents novel frameworks for thinking about problems
- Creates mental models others will reference
- Introduces concepts that simplify complex ideas
- Generates quotable principles or heuristics

Medium memetic potential (4-6):
- Applies existing frameworks in new ways
- Synthesizes ideas into useful combinations
- Provides memorable examples or analogies
- Clarifies existing concepts effectively

Low memetic potential (1-3):
- Discusses existing ideas without new framing
- Commentary on established concepts
- Personal reflections without generalizable insights
- Technical details without broader applicability

Post content to grade:
{title}

{post_text}

Consider:
- Does this introduce new vocabulary or concepts?
- Will other posts likely reference this framework?
- Are there quotable insights or principles?
- Does it create tools for thinking about problems?
- Will this change how EAs discuss this topic?

Respond with JSON:
{{
    "memetic_potential_score": <int 1-10>,
    "explanation": "<brief explanation noting any new terms, frameworks, or concepts introduced>"
}}
"""


def compute_memetic_potential_v2(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> MemeticPotentialV2:
    """Compute memetic potential score for a post.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        MemeticPotentialV2 metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    prompt = PROMPT_MEMETIC_POTENTIAL.format(
        title=post.title,
        post_text=post_text
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    result = parse_json_with_repair(raw_content)
    
    return MemeticPotentialV2(
        post_id=post.post_id,
        memetic_potential_score=result["memetic_potential_score"],
        explanation=result["explanation"]
    )