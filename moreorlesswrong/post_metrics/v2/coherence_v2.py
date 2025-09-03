from typing import Literal
from pydantic import BaseModel
import json

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair


class CoherenceV2(BaseModel):
    post_id: str
    coherence_score: int  # 1-10 overall coherence score
    explanation: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "CoherenceV2"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["coherence_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "coherence_score": "Coherence Score"
        }


PROMPT_COHERENCE = """Evaluate the COHERENCE of this EA Forum post.

"Are the claims expressed in the post mutually consistent?"

Coherence is the virtue of having one's attitudes be mutually compatible. A system that expresses both propositions p and ~p is expressing incompatible attitudes.

Example of incoherence: If I bet on a race between A, B, and C (no ties), believing their chances are 20/100, 40/100, and 50/100 respectively (adding up to 110/100), I will necessarily have overpaid and lost money.

Note: One can be coherent yet be inaccurate. A conspiracy theorist might be perfectly coherent in a fundamentally inaccurate worldview where everything "fits together".

First, list out the top central claims made by the post. Then, assess whether these claims are mutually consistent. Finally, assess the coherence of the post as a whole.

Post title: {title}

Post content to grade:
{post_text}

Rubric:
Grade on a 1-10 scale for coherence:
- 1-2: Severely incoherent, multiple contradictions, expresses p and ~p
- 3-4: Notable inconsistencies, views sanction buying a Dutch Book
- 5-6: Mostly coherent with some incompatible attitudes
- 7-8: Coherent with minor issues
- 9-10: Perfectly coherent, all attitudes mutually compatible

Respond with JSON:
{{
    "coherence_score": <int 1-10 overall coherence>,
    "explanation": "<brief explanation of coherence assessment>"
}}
"""


def compute_coherence_v2(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> CoherenceV2:
    """Compute coherence scores for a post using epistemic virtue principles.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        CoherenceV2 metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    prompt = PROMPT_COHERENCE.format(
        title=post.title,
        post_text=post_text
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    result = parse_json_with_repair(raw_content)
    
    return CoherenceV2(
        post_id=post.post_id,
        coherence_score=result["coherence_score"],
        explanation=result["explanation"]
    )