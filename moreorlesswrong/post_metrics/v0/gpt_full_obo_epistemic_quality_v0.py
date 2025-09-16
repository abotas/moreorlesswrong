from typing import Literal
from pydantic import BaseModel

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair


class GptFullOBOEpistemicQualityV0(BaseModel):
    post_id: str
    epistemic_quality_score: int  # 1-10 overall epistemic quality score
    explanation: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "GptFullOBOEpistemicQualityV0"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["epistemic_quality_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "epistemic_quality_score": "GPT-Full OBO Epistemic Quality Score"
        }


PROMPT_GPT_OBO_EPISTEMIC = """Consider the holistic epistemic quality of this EA forum post. Rate on a scale 1-10.
```
{title}
{content}
```
Return your response in this exact JSON format:
{{
    "explanation": "Brief explanation of why you gave this rating",
    "epistemic_quality_score": <score from 1-10>
}}
"""


def compute_gpt_full_obo_epistemic_quality_v0(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini",
    bypass_synthesizer: bool = False,
    n_related_posts: int = 5,
) -> GptFullOBOEpistemicQualityV0:
    """Compute GPT-Full OBO epistemic quality score for a post.
    
    Args:
        post: The post to evaluate
        model: Ignored - always uses gpt-5
        bypass_synthesizer: Ignored - V0 metrics don't use synthesis
        n_related_posts: Ignored - V0 metrics don't use related posts
        
    Returns:
        GptFullOBOEpistemicQualityV0 metric object
    """
    prompt = PROMPT_GPT_OBO_EPISTEMIC.format(
        title=post.title,
        content=post.markdown_content
    )
    
    response = client.chat.completions.create(
        model="gpt-5",  # Hardcoded to gpt-5
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    result = parse_json_with_repair(raw_content)
    
    return GptFullOBOEpistemicQualityV0(
        post_id=post.post_id,
        epistemic_quality_score=result["epistemic_quality_score"],
        explanation=result["explanation"]
    )