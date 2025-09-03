from typing import Literal
from pydantic import BaseModel
import json

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair


class CooperativenessV2(BaseModel):
    post_id: str
    cooperativeness_score: int  # 1-10 overall cooperativeness score
    explanation: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "CooperativenessV2"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["cooperativeness_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "cooperativeness_score": "Cooperativeness Score"
        }


PROMPT_COOPERATIVENESS = """Evaluate the COOPERATIVENESS of this EA Forum post.

"Is the post providing information in a way that improves the epistemic situation of the recipient?"

Cooperativeness means taking responsibility for how to convey the right information in a way that is beneficial to the recipient, putting them in the best epistemic situation they can. This is effectively causing appropriate understanding in an intended subject.

Being cooperative involves being:
- Intelligible: Presenting information in a way that is accessible to the recipient at low cost
- Explanatory: Providing context that gives recipients autonomy to assess answers and related inquiries themselves
- Epistemically aligned: Not selectively providing information in a manipulative manner

Example of failure: Media outlets that provide disproportional reporting on outlandish events like terror attacks, causing inaccurate beliefs about the prevalence of those events.
Example of success: Explaining Newtonian physics differently to a 5-year-old, a high school student, and a college student majoring in physics.


Post content to grade:
{title}

{post_text}

Rubric:
Grade on a 1-10 scale for cooperativeness to *EA forum readership*:
- 1-2: Provides information that can only be decoded at great cost or appears to attempt to manipulate the recipient
- 3-4: Poor presentation, difficult to understand, potentially manipulative
- 5-6: Moderate cooperativeness, some issues with accessibility or completeness
- 7-8: Good presentation, helpful context, epistemically aligned
- 9-10: Excellent epistemic cooperation, puts recipients in best epistemic situation

Respond with JSON:
{{
    "cooperativeness_score": <int 1-10 overall cooperativeness>,
    "explanation": "<brief explanation of cooperativeness assessment>"
}}
"""


def compute_cooperativeness_v2(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> CooperativenessV2:
    """Compute cooperativeness scores for a post using epistemic virtue principles.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        CooperativenessV2 metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    prompt = PROMPT_COOPERATIVENESS.format(
        title=post.title,
        post_text=post_text
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    result = parse_json_with_repair(raw_content)
    
    return CooperativenessV2(
        post_id=post.post_id,
        cooperativeness_score=result["cooperativeness_score"],
        explanation=result["explanation"]
    )