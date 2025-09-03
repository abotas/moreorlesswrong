from typing import Literal
from pydantic import BaseModel

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair


class RobustnessV2(BaseModel):
    post_id: str
    robustness_score: int  # 1-10
    actionable_feedback: str  # The feedback generated in step 1
    improvement_potential: str  # Explanation of how much better the post could be
    
    @classmethod
    def metric_name(cls) -> str:
        return "RobustnessV2"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["robustness_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "robustness_score": "Robustness Score"
        }


PROMPT_STEP1_POST_FEEDBACK = """You are a critical reviewer highlighting mistakes on an EA forum post.

Identify the 1-3 biggest weaknesses in this post. We're looking only for any large mistakes the author has made. Consider:
- Large reasoning errors
- Key dubious assumptions that are far from obvious for readers of EA forum
- Overlooked highly reasonable counterarguments

We're not interested in whether you believe the arguments the post is making are correct. But we're interested in whether there are any major mistakes, 'own goals' that the author would be embarrassed to have made if they're brought to the author's attention.

Provide specific, actionable 1-2 points of the most important feedback that the author should consider before publishing.
Keep in mind author and reader time is precious - imagine there's a steep regularization penalty to post length so feedback that would lengthen the post should be carefully considered.

Post content:
{title}

{post_text}

Respond with valid JSON containing your feedback:
{{"feedback": "<your 1-3 key points of actionable feedback that would improve this post>"}}
"""


PROMPT_STEP2_POST_EVALUATE = """You are evaluating feedback on an EA forum post. Your task is to 
critically rate the feedback. You should be assessing to what extent the feedback identifies what amount to major mistakes made by the author. 
To what extent does the feedback identify 'own goals' that the author would be embarrassed to have made if they're brought to the author's attention.

Keep in mind also that reader time is precious - imagine there's a steep regularization penalty to post length so feedback that would lengthen the post must be carefully considered.

Content to grade:
Post:
`{title}

{post_text}`

Potential mistakes identified:
`{feedback}`

Rubric:
Grade on a 1-10 scale for how critical the identified issues are.
- 1-2: The identified issues amount to mild mistakes. It's unclear the author would agree these are mistakes that must be corrected before publishing. Fixing them would require lengthening the post significantly.
- 3-4: The identified issues amount to moderate mistakes. The author may agree these are mistakes that must be corrected before publishing. Fixing them does not require lengthening the post a lot.
- 5-6: The identified issues amount to large mistakes. The author would likely agree these are mistakes that must be corrected before publishing. Fixing them does not require lengthening the post significantly.
- 7-8: The identified issues amount to very lage mistakes. The author would surely agree these are mistakes that must be corrected before publishing. Fixing them does not require lengthening the post much.
- 9-10: The identified issues amount to enormous mistakes. The author would likely be hugely embarrassed if they were made aware of these mistakes. Fixing them does not require lengthening the post.

Respond with valid JSON:
{{"feedback_score": <int 1-10>, "improvement_potential": "<brief explanation of why this score>"}}
"""


def compute_robustness_v2(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> RobustnessV2:
    """Compute robustness score for a post using two-step evaluation.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        RobustnessV2 metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    # Step 1: Generate actionable feedback
    prompt_step1 = PROMPT_STEP1_POST_FEEDBACK.format(
        title=post.title,
        post_text=post_text
    )
    
    response_step1 = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_step1}]
    )
    
    raw_content_step1 = response_step1.choices[0].message.content
    result_step1 = parse_json_with_repair(raw_content_step1)
    feedback = result_step1["feedback"]
    
    # Step 2: Evaluate the usefulness of the feedback
    prompt_step2 = PROMPT_STEP2_POST_EVALUATE.format(
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
    return RobustnessV2(
        post_id=post.post_id,
        robustness_score=11 - int(result["feedback_score"]),
        actionable_feedback=feedback,
        improvement_potential=result["improvement_potential"]
    )