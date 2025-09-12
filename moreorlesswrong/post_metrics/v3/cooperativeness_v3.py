from typing import Literal
from pydantic import BaseModel
import json

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair
from db import get_n_most_recent_posts_in_same_cluster
from synthesizer import synthesize_context


class CooperativenessV3(BaseModel):
    post_id: str
    cooperativeness_score: int  # 1-10 overall cooperativeness score
    analysis_of_cooperativeness: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "CooperativenessV3"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["cooperativeness_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "cooperativeness_score": "Cooperativeness Score"
        }


# Extract the evaluation criteria for the synthesizer
COOPERATIVENESS_EVALUATION_CRITERIA = """Evaluate the COOPERATIVENESS of this EA Forum post.

"Is the post providing information in a way that improves the epistemic situation of the recipient?"

Cooperativeness means taking responsibility for how to convey the right information in a way that is beneficial to the recipient, putting them in the best epistemic situation they can. This is effectively causing appropriate understanding in an intended subject.

Being cooperative involves being:
- Intelligible: Presenting information in a way that is accessible to the recipient at low cost
- Explanatory: Providing context that gives recipients autonomy to assess answers and related inquiries themselves
- Epistemically aligned: Not selectively providing information in a manipulative manner

Example of failure: Media outlets that provide disproportional reporting on outlandish events like terror attacks, causing inaccurate beliefs about the prevalence of those events.
Example of success: Explaining Newtonian physics differently to a 5-year-old, a high school student, and a college student majoring in physics.

Rubric:
Grade on a 1-10 scale for cooperativeness to *EA forum readership*:
- 1-2: Provides information that can only be decoded at great cost or appears to attempt to manipulate the recipient
- 3-4: Poor presentation, difficult to understand, potentially manipulative
- 5-6: Moderate cooperativeness, some issues with accessibility or completeness
- 7-8: Good presentation, helpful context, epistemically aligned
- 9-10: Excellent epistemic cooperation, puts recipients in best epistemic situation

You'll first provide an analysis of the cooperativeness of the post using this rubric: How clearly does the post present its thesis supporting argumentation? Is an appropriate level of evidence, explanation, general context provided? Is it presented at an appropriate level of complexity for EA forum readership? Is the author epistemically aligned with the EA forum readership? Does the post cause the readership to understand something of value

Then you'll provide a score on a 1-10 scale.
"""


PROMPT_COOPERATIVENESS_V3 = """{evaluation_criteria}

Synthesis agent compiled some potentially useful context from recent, related posts:
{synthesized_info}

Post content to grade:
```
{title}
{post_text}
```

Respond with JSON:
```json{{
    "analysis_of_cooperativeness": "<Analyze the cooperativeness of the post>",
    "cooperativeness_score": <int 1-10 overall cooperativeness>,
}}
```
"""


def compute_cooperativeness_v3(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini",
) -> CooperativenessV3:
    """Compute cooperativeness scores for a post with synthesized information from related posts.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        CooperativenessV3 metric object
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
        metric_name="Cooperativeness",
        metric_evaluation_prompt=COOPERATIVENESS_EVALUATION_CRITERIA,
        model=model
    )

    prompt = PROMPT_COOPERATIVENESS_V3.format(
        evaluation_criteria=COOPERATIVENESS_EVALUATION_CRITERIA,
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
    
    return CooperativenessV3(
        post_id=post.post_id,
        cooperativeness_score=result["cooperativeness_score"],
        analysis_of_cooperativeness=result["analysis_of_cooperativeness"]
    )