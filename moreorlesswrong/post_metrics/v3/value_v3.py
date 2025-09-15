from typing import Literal, Optional
from pydantic import BaseModel
import json

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair
from db import get_n_most_recent_posts_in_same_cluster
from prev_post_synthesizer import synthesize_context
from raw_context_formatter import format_raw_related_posts


class ValueV3(BaseModel):
    post_id: str
    value_score: int  # 1-10 overall value score
    analysis: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "ValueV3"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["value_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "value_score": "Value Score"
        }


# Extract the evaluation criteria for the synthesizer
VALUE_EVALUATION_CRITERIA = """Evaluate the VALUE of this post to EA forum readership.

"Would the information provided make a difference to readership if it were accurate?"

Virtuous epistemic posts say things that matter and aim to communicate valuable information.

Before evaluating value, consider:
- Does the information make a difference to an inquiry about the world?
- Does the inquiry make a difference to readership?

Examples of low Value:
- "Water is wet" - makes no difference to any realistic inquiry
- "David Hume died at 3:51pm" - while extremely informative (rules out he died at any other time), makes no difference to inquiries that matter given my interests
- "Will McAskill wrote a book outlining his moral philosophy" - this is informative, but does not make a difference to the readership who will be familiar with his work

Rubric:
Grade on a 1-10 scale for value to EA forum readership:
- 1-2: Says a series of true platitudes instead of providing important information
- 3-4: Limited value, information doesn't matter to important inquiries  
- 5-6: Moderate value, somewhat difference-making
- 7-8: High value, addresses inquiries that matter to agents
- 9-10: Exceptional value, information would make crucial differences to decisions

First you'll identify the main thesis of the post and any key claims or arguments or corollaries. You'll assess the value of each of these. Then assess the overall value of the post according to the rubric. 
Then you'll provide a score on a 1-10 scale.
"""

SYNTHESIZER_FOCUS_AREA = """Look for:
- Whether this post addresses questions or concerns raised in previous discussions
- If the post builds meaningfully on prior work or fills identified knowledge gaps  
- How the post's claims or evidence complement or contradict previous findings
- Whether the timing and context of previous posts might make this post more or less valuable right now"""


PROMPT_VALUE_V3 = """{evaluation_criteria}

Synthesis agent compiled some potentially useful context from recent, related posts:
```
{synthesized_info}
```

Post content to grade:
```
{title}
{post_text}
```

Respond with JSON:
```json{{
    "analysis": "<Identification of main thesis and key claims or arguments or corollaries. Assessment of value of each. Overall value assessment.>"
    "value_score": <int 1-10 overall value>,
}}
```
"""


def compute_value_v3(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini",
    bypass_synthesizer: bool = False,
    n_related_posts: int = 5,
) -> ValueV3:
    """Compute value score for a post with synthesized information from related posts.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        bypass_synthesizer: Whether to bypass synthesizer and use raw related posts (default: False)
        
    Returns:
        ValueV3 metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    related_posts = get_n_most_recent_posts_in_same_cluster(
        post_id=post.post_id,
        cluster_cardinality=12,
        n=n_related_posts
    )
    if len(related_posts) < n_related_posts:
        related_posts2 = get_n_most_recent_posts_in_same_cluster(
            post_id=post.post_id,
            cluster_cardinality=5,
            n=n_related_posts*2
        )
        postids = [p.post_id for p in related_posts]
        related_posts = related_posts + [p for p in related_posts2 if p.post_id not in postids]
        related_posts = related_posts[:n_related_posts]

    # Use either synthesizer or raw related posts formatting
    if bypass_synthesizer:
        synthesized_info = format_raw_related_posts(related_posts)
    else:
        synthesized_info = synthesize_context(
            new_post=post,
            previous_posts=related_posts,
            metric_name="Value",
            metric_evaluation_prompt=VALUE_EVALUATION_CRITERIA,
            model=model,
            synthesizer_focus_area=SYNTHESIZER_FOCUS_AREA
        )

    prompt = PROMPT_VALUE_V3.format(
        evaluation_criteria=VALUE_EVALUATION_CRITERIA,
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
    
    return ValueV3(
        post_id=post.post_id,
        value_score=result["value_score"],
        analysis=result["analysis"]
    )