from typing import Literal
from pydantic import BaseModel
import json

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair


class TruthfulnessV2(BaseModel):
    post_id: str
    truthfulness_score: int  # 1-10 overall truthfulness score
    explanation: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "TruthfulnessV2"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["truthfulness_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "truthfulness_score": "Truthfulness Score"
        }


PROMPT_TRUTHFULNESS = """Evaluate the TRUTHFULNESS of this EA Forum post.

"Is the author of the post providing truthful information?"

Consider:
- Do the statements correspond to how things are?
- Do the statements omit pertinent information?
- Are the claims true rather than false?
- Would a virtuous agent with perfect knowledge agree with these statements?


Post content to grade:
{title}
{post_text}

Rubric:
Grade on a 1-10 scale for truthfulness:
- 1-2: Frequently spouts falsehoods
- 3-4: Contains significant inaccuracies or omits some central information
- 5-6: Mostly accurate with some false claims or pertinent omissions
- 7-8: Accurate with minor issues
- 9-10: Highly accurate, statements correspond to how things are

Respond with JSON:
{{
    "truthfulness_score": <int 1-10 overall truthfulness>,
    "explanation": "<brief explanation of truthfulness assessment>"
}}
"""


def compute_truthfulness_v2(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> TruthfulnessV2:
    """Compute truthfulness scores for a post using epistemic virtue principles.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        TruthfulnessV2 metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    prompt = PROMPT_TRUTHFULNESS.format(
        title=post.title,
        post_text=post_text
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    result = parse_json_with_repair(raw_content)
    
    return TruthfulnessV2(
        post_id=post.post_id,
        truthfulness_score=result["truthfulness_score"],
        explanation=result["explanation"]
    )