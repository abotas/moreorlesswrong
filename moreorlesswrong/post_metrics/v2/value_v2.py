from typing import Literal
from pydantic import BaseModel
import json

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair


class ValueV2(BaseModel):
    post_id: str
    value_score: int  # 1-10 overall value score
    explanation: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "ValueV2"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["value_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "value_score": "Value Score"
        }


PROMPT_VALUE = """Evaluate the VALUE of this post to EA forum readership.

"Would the information provided make a difference to readership if it were accurate?"

Virtuous epistemic posts say things that matter and aim to communicate valuable information.

Before evaluating value, consider:
- Does the information make a difference to an inquiry about the world?
- Does the inquiry make a difference to readership?

Examples of low Value:
- "Water is wet" - makes no difference to any realistic inquiry
- "David Hume died at 3:51pm" - while extremely informative (rules out he died at any other time), makes no difference to inquiries that matter given my interests
- "Will McAskill wrote a book outlining his moral philosophy" - this is informative, but does not make a difference to the readership who will be familiar with his work

Post content to grade:
{title}
{post_text}

Rubric:
Grade on a 1-10 scale for value to EA forum readership:
- 1-2: Says a series of true platitudes instead of providing important information
- 3-4: Limited value, information doesn't matter to important inquiries  
- 5-6: Moderate value, somewhat difference-making
- 7-8: High value, addresses inquiries that matter to agents
- 9-10: Exceptional value, information would make crucial differences to decisions

Respond with JSON:
{{
    "value_score": <int 1-10 overall value>,
    "explanation": "<brief explanation of value assessment>"
}}
"""


def compute_value_v2(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> ValueV2:
    """Compute value scores for a post using epistemic virtue principles.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        ValueV2 metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    prompt = PROMPT_VALUE.format(
        title=post.title,
        post_text=post_text
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    result = parse_json_with_repair(raw_content)
    
    return ValueV2(
        post_id=post.post_id,
        value_score=result["value_score"],
        explanation=result["explanation"]
    )