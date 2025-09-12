from typing import Literal
from pydantic import BaseModel

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair
from db import get_n_most_recent_posts_in_same_cluster
from synthesizer import synthesize_context


class RobustnessV3(BaseModel):
    post_id: str
    robustness_score: int  # 1-10
    actionable_feedback: str  # The feedback generated in step 1
    improvement_potential: str  # Explanation of how much better the post could be
    
    @classmethod
    def metric_name(cls) -> str:
        return "RobustnessV3"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["robustness_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "robustness_score": "Robustness Score"
        }


# Extract the evaluation criteria for the synthesizer
ROBUSTNESS_EVALUATION_CRITERIA = """Evaluate the ROBUSTNESS of an EA forum post.

Robustness is measured by identifying major mistakes and weaknesses in the post. We're looking for:
- Large reasoning errors
- Key dubious assumptions that are far from obvious for readers of EA forum
- Overlooked highly reasonable counterarguments

We're interested in whether there are any major mistakes, 'own goals' that the author would be embarrassed to have made if brought to their attention.

The robustness score reflects how free the post is from such critical errors:
- High robustness (9-10): Post is virtually free from major mistakes
- Good robustness (7-8): Only minor issues that wouldn't embarrass the author
- Moderate robustness (5-6): Some notable mistakes that should be corrected
- Poor robustness (3-4): Several significant errors that undermine the argument
- Very poor robustness (1-2): Major embarrassing mistakes throughout"""


PROMPT_STEP1_POST_FEEDBACK_V3 = """You are a critical reviewer highlighting mistakes on an EA forum post.

Synthesis agent compiled some potentially useful context from recent, related posts:
{synthesized_info}

Identify the 1-3 biggest weaknesses in this post. We're looking only for any large mistakes the author has made. Consider:
- Large reasoning errors
- Key dubious assumptions that are far from obvious for readers of EA forum
- Overlooked highly reasonable counterarguments

We're not interested in whether you believe the arguments the post is making are correct. But we're interested in whether there are any major mistakes, 'own goals' that the author would be embarrassed to have made if they're brought to the author's attention.

Provide specific, actionable 1-2 points of the most important feedback that the author should consider before publishing.
Keep in mind author and reader time is precious - imagine there's a steep regularization penalty to post length so feedback that would lengthen the post should be carefully considered.

Post content:
```
{title}
{post_text}
```

Respond with valid JSON containing your feedback:
```json{{
    "feedback": "<your 1-3 key points of actionable feedback that would improve this post>"
}}
```
"""


PROMPT_STEP2_POST_EVALUATE_V3 = """You are evaluating feedback on an EA forum post. Your task is to critically rate the feedback. 
You should be assessing to what extent the feedback identifies what amount to major mistakes made by the author. 
To what extent does the feedback identify 'own goals' that the author would be embarrassed to have made if they're brought to the author's attention.

Keep in mind also that reader time is precious - imagine there's a steep regularization penalty to post length so feedback that would lengthen the post must be carefully considered.

Content to grade:
Post:
```
{title}
{post_text}
```

Potential mistakes identified:
`{feedback}`

Rubric:
Grade on a 1-10 scale for how critical the identified issues are.
- 1-2: The identified issues amount to mild mistakes. It's unclear the author would agree these are mistakes that must be corrected before publishing. Fixing them would require lengthening the post significantly.
- 3-4: The identified issues amount to moderate mistakes. The author may agree these are mistakes that must be corrected before publishing. Fixing them does not require lengthening the post a lot.
- 5-6: The identified issues amount to large mistakes. The author would likely agree these are mistakes that must be corrected before publishing. Fixing them does not require lengthening the post significantly.
- 7-8: The identified issues amount to very large mistakes. The author would surely agree these are mistakes that must be corrected before publishing. Fixing them does not require lengthening the post much.
- 9-10: The identified issues amount to enormous mistakes. The author would likely be hugely embarrassed if they were made aware of these mistakes. Fixing them does not require lengthening the post.

Respond with valid JSON:
```json{{
    "improvement_potential": "<brief explanation of why this score>",
    "feedback_score": <int 1-10>
}}
```
"""


def compute_robustness_v3(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini",
) -> RobustnessV3:
    """Compute robustness score for a post with synthesized information from related posts.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        RobustnessV3 metric object
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
        metric_name="Robustness",
        metric_evaluation_prompt=ROBUSTNESS_EVALUATION_CRITERIA,
        model=model
    )
    
    # Step 1: Generate actionable feedback with synthesized context
    prompt_step1 = PROMPT_STEP1_POST_FEEDBACK_V3.format(
        title=post.title,
        post_text=post_text,
        synthesized_info=synthesized_info
    )
    
    response_step1 = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_step1}]
    )
    
    raw_content_step1 = response_step1.choices[0].message.content
    result_step1 = parse_json_with_repair(raw_content_step1)
    feedback = result_step1["feedback"]
    
    # Step 2: Evaluate the usefulness of the feedback
    prompt_step2 = PROMPT_STEP2_POST_EVALUATE_V3.format(
        title=post.title,
        feedback=feedback,
        post_text=post_text
    )
    
    response_step2 = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_step2}]
    )
    raw_content = response_step2.choices[0].message.content
    result = parse_json_with_repair(raw_content)
    assert int(result["feedback_score"]) in range(1, 11)
    return RobustnessV3(
        post_id=post.post_id,
        robustness_score=11 - int(result["feedback_score"]),
        actionable_feedback=feedback,
        improvement_potential=result["improvement_potential"]
    )