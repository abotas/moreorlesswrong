from typing import Literal
from pydantic import BaseModel
import json

from models import Post, Claim
from llm_client import client


class Robustness(BaseModel):
    post_id: str
    claim_id: str
    robustness_score: int  # 1-10
    actionable_feedback: str  # The feedback generated in step 1
    improvement_potential: str  # Explanation of how much better the claim could be

    @classmethod
    def from_model_response_json(cls, response: str) -> "Robustness":
        return cls(**json.loads(response))
    
    @classmethod
    def metric_name(cls) -> str:
        return "Robustness"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["robustness_score"]


PROMPT_STEP1_FEEDBACK = """You are a critical reviewer providing actionable feedback on a particular claim from an EA forum post and how it is supported.

Identify the 1-3 most important ways this claim and its support could be improved. Consider:
- Reasoning errors
- Key assumptions that are far from obvious for readers of EA forum
- Overlooked highly plausible counterarguments

Provide specific, actionable 1-3 points of the most important feedback for this claim and its support that the author should consider before they publish.
Keep in mind author and reader time is precious - imagine there's a regularization penalty to post length so feedback that would lengthen the post should be
carefully considered.

The claim:
"{claim}"

Full post for context:
{title}
{post_text}

Respond with valid JSON containing your feedback:
{{"feedback": "<your 1-3 key points of actionable feedback that would improve this claim>"}}
"""


PROMPT_STEP2_EVALUATE = """You are evaluating feedback on a particular claim made in an EA forum post. Your task is to 
critically rate how useful the feedback is for improving the substance of the claim and its support in the post, while bearing in mind
that being concise is a virtue. Readers' time is precious. 
Original claim: "{claim}"

is supported by this post:
{post_text}

Proposed feedback:
{feedback}

Rate on a 1-10 scale how useful the feedback is for improving the claim and its support in this post. We have very high standards for this metric:
- 1: Feedback is irrelevant or would make the claim worse
- 3: Moderate improvements, addressing the feedback would likely improve the claim and support. Or large improvements however addressing the feedback would require lengthening the post significantly. Or the feedback is relevant but the post is a link post with supporting text intentionally elsewhere.
- 5: Large improvements, addresses key weaknesses in the claim or its support. And addressing them would not significantly lengthen the post.
- 7: Critical improvements, without addressing the feedback, the claim missing critical support.
- 10: Critical improvements, without addressing the feedback, the claim is clearly wrong.

Respond with valid JSON:
{{"feedback_score": <int 1-10>, "improvement_potential": "<brief explanation of why this score>"}}
"""


def compute_robustness(
    claim: Claim,
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> Robustness:
    """Compute robustness score using two-step evaluation.
    
    Args:
        claim: The claim to evaluate
        post: The post containing the claim
        model: The model to use for evaluation
        
    Returns:
        Robustness metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    # Step 1: Generate actionable feedback
    prompt_step1 = PROMPT_STEP1_FEEDBACK.format(
        title=post.title,
        claim=claim.claim,
        post_text=post_text
    )
    
    response_step1 = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_step1}]
    )
    
    raw_content_step1 = response_step1.choices[0].message.content
    
    # Strip potential markdown formatting
    if raw_content_step1.startswith("```"):
        raw_content_step1 = raw_content_step1.split("```")[1]
        if raw_content_step1.startswith("json"):
            raw_content_step1 = raw_content_step1[4:]
        raw_content_step1 = raw_content_step1.strip()
    
    result_step1 = json.loads(raw_content_step1)
    feedback = result_step1["feedback"]
    
    # Step 2: Evaluate the usefulness of the feedback (could use different model)
    prompt_step2 = PROMPT_STEP2_EVALUATE.format(
        claim=claim.claim,
        feedback=feedback,
        post_text=post_text
    )
    # print('\n----robustness prompt_step2----')
    # print(prompt_step2)
    response_step2 = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_step2}]
    )
    raw_content = response_step2.choices[0].message.content
    # print('\n')
    # print(raw_content)
    # print('----robustness response_step2----\n')
    
    # Strip potential markdown formatting
    if raw_content.startswith("```"):
        raw_content = raw_content.split("```")[1]
        if raw_content.startswith("json"):
            raw_content = raw_content[4:]
        raw_content = raw_content.strip()
    
    result = json.loads(raw_content)
    assert int(result["feedback_score"]) in range(1, 11)
    return Robustness(
        post_id=post.post_id,
        claim_id=claim.claim_id,
        robustness_score=11 - int(result["feedback_score"]),
        actionable_feedback=feedback,
        improvement_potential=result["improvement_potential"]
    )