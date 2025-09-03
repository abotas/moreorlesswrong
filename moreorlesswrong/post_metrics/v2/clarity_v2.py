from typing import Literal
from pydantic import BaseModel
import json

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair


class ClarityV2(BaseModel):
    post_id: str
    clarity_score: int  # 1-10 overall clarity score
    explanation: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "ClarityV2"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["clarity_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "clarity_score": "Clarity Score"
        }


PROMPT_CLARITY = """Evaluate the CLARITY of this EA Forum post.

"How easy is it to understand the information that the post provides?"

The Clarity of a communicated message is how costly it is for the intended recipient to extract the information from the message. 
We can communicate messages in very complicated ways, even if they carry the exact same information. For example, the sentences 'The number of sheep is the twelfth prime number' and 'The number of sheep is 37' express the same content, but one is costlier to understand.

Clarity can trade off against Precision, Accuracy, and Value. When explaining something to a young child, you might have to omit details that would be valuable if the child could understand it.

Post content to grade:
{title}

{post_text}

Rubric:
Grade on a 1-10 scale for clarity to *EA forum readership*:
- 1-2: Very unclear, high cognitive cost to extract information that could be presented more clearly
- 3-4: Difficult to understand, messages require significant unnecessary processing
- 5-6: Decent clarity, but some unnecessary cognitive processing would be required for typical EA readership to understand
- 7-8: Good clarity, but some unnecessary cognitive processing would be required for typical EA readership to understand
- 9-10: Optimally clear, minimal cognitive cost beyond what is absolutely necessary to transmit the information

Respond with JSON:
{{
    "clarity_score": <int 1-10 overall clarity>,
    "explanation": "<brief explanation of clarity assessment>"
}}
"""


def compute_clarity_v2(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> ClarityV2:
    """Compute clarity scores for a post using epistemic virtue principles.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        ClarityV2 metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    prompt = PROMPT_CLARITY.format(
        title=post.title,
        post_text=post_text
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    result = parse_json_with_repair(raw_content)
    
    return ClarityV2(
        post_id=post.post_id,
        clarity_score=result["clarity_score"],
        explanation=result["explanation"]
    )