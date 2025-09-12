from typing import Literal
from pydantic import BaseModel
import json

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair
from db import get_n_most_recent_posts_in_same_cluster
from synthesizer import synthesize_context


class PrecisionV3(BaseModel):
    post_id: str
    precision_score: int  # 1-10 overall precision score
    analysis: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "PrecisionV3"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["precision_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "precision_score": "Precision Score"
        }


# Extract the evaluation criteria for the synthesizer
PRECISION_EVALUATION_CRITERIA = """Evaluate the PRECISION of this post for EA forum readership.

Precision is a measure of how informative the claims of a post are, in the sense of Shannon information. The higher precision, the more possible ways the world could be that the post rules out.

Being appropriately Precise requires saying the most informative thing you can that brings marginal value, without sacrificing truthfulness.

Although more precise information typically increases value, sometimes marginal information only makes a difference to inquiries we don't care about. For example, if I am choosing a college, I don't need to be informed about the building materials of the registrar's office.

You'll first provide an analysis moving section by section through the post, analyzing the appropriateness of the precision in each section. Then you'll consider the overall precision of the post. 

Then you'll provide a score on a 1-10 scale.

Rubric:
Grade on a 1-10 scale for OPTIMAL precision to *EA forum readership*:
- 1-2: Very poor precision balance - either too vague (platitudes) OR too detailed (superfluous minutiae)
- 3-4: Poor precision calibration - either underspecified OR cluttered with irrelevant detail
- 5-6: Moderate precision - some sections appropriately precise, others miss the mark
- 7-8: Well-calibrated precision - appropriately informative for the context and audience
- 9-10: Optimally precise - perfect balance of informativeness and relevance, neither too vague nor unnecessarily detailed
"""


PROMPT_PRECISION_V3 = """{evaluation_criteria}

Synthesis agent compiled some potentially useful context from recent, related posts:
{synthesized_info}

Post content to grade:
```
{title}
{post_text}
```

Respond with JSON:
```json{{
    "analysis": "<Precision assessment>"
    "precision_score": <int 1-10 overall precision>,
}}
```
"""


def compute_precision_v3(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini",
) -> PrecisionV3:
    """Compute precision score for a post with synthesized information from related posts.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        PrecisionV3 metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    # Get 5 most recent posts from the same cluster-5
    related_posts = get_n_most_recent_posts_in_same_cluster(
        post_id=post.post_id,
        cluster_cardinality=5,
        n=5
    )

    synthesized_info = synthesize_context(
        new_post=post,
        previous_posts=related_posts,
        metric_name="Precision",
        metric_evaluation_prompt=PRECISION_EVALUATION_CRITERIA,
        model=model
    )

    prompt = PROMPT_PRECISION_V3.format(
        evaluation_criteria=PRECISION_EVALUATION_CRITERIA,
        title=post.title,
        post_text=post_text,
        synthesized_info=synthesized_info
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    result = parse_json_with_repair(raw_content)
    
    return PrecisionV3(
        post_id=post.post_id,
        precision_score=result["precision_score"],
        analysis=result["analysis"]
    )