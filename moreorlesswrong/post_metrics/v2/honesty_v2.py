from typing import Literal
from pydantic import BaseModel
import json

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair


class HonestyV2(BaseModel):
    post_id: str
    honesty_score: int  # 1-10 overall honesty score
    explanation: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "HonestyV2"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["honesty_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "honesty_score": "Honesty Score"
        }


PROMPT_HONESTY = """Evaluate the HONESTY of this EA Forum post.

"Is the author of the post optimizing for goal that is distinct from the goals of EA forum readership?"

Consider whether it seems that the author has ulterior motives beyond providing accurate information on the stated topic.

An example of such deceptive behavior is providing selective information that is itself accurate, but that causes an inaccurate representation of the world in the recipient. For example, a post might provide selective research reports on a topic supporting only one side. This information might be valuable, but not as valuable as a more complete picture would be.
Another would be a post that claims to support using some new tool 


Post content to grade:
{title}

{post_text}

Rubric:
Grade on a 1-10 scale for honesty:
- 1-2: Moderate to severe selective presentation issues, or clear conflicts of interest or ulterior motives from the author beyond providing accurate information
- 3-4: Potential moderate-to-severe conflicts of interests, potential selective presentation issues.
- 5-6: Minor conflicts of interests, potential selective presentation issues.
- 7-8: Generally honest with minor selectivity, mostly complete
- 9-10: Clearly honest, transparently optimizing for epistemic virtue

Respond with JSON:
{{
    "honesty_score": <int 1-10 overall honesty>,
    "explanation": "<brief explanation of honesty assessment>"
}}
"""


def compute_honesty_v2(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> HonestyV2:
    """Compute honesty scores for a post using epistemic virtue principles.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        HonestyV2 metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    prompt = PROMPT_HONESTY.format(
        title=post.title,
        post_text=post_text
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    result = parse_json_with_repair(raw_content)
    
    return HonestyV2(
        post_id=post.post_id,
        honesty_score=result["honesty_score"],
        explanation=result["explanation"]
    )