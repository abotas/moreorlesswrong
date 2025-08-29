from typing import Literal
from pydantic import BaseModel

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair


class PostRobustness(BaseModel):
    post_id: str
    robustness_score: int  # 1-10
    actionable_feedback: str  # The feedback generated in step 1
    improvement_potential: str  # Explanation of how much better the post could be
    
    @classmethod
    def metric_name(cls) -> str:
        return "PostRobustness"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["robustness_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "robustness_score": "Robustness Score"
        }


PROMPT_STEP1_POST_FEEDBACK = """You are a critical reviewer providing actionable feedback on an EA forum post.

Identify the 1-3 biggest weaknesses in this post. We're looking for any large mistakes the author has made. Consider:
- Large reasoning errors
- Key dubiousassumptions that are far from obvious for readers of EA forum
- Overlooked highly plausible counterarguments

We're not interested in whether you believe the arguments the post is making are correct. But we're interested in whether there are any major mistakes
made by the author. Own goals.

Provide specific, actionable 1-3 points of the most important feedback that the author should consider before publishing.
Keep in mind author and reader time is precious - imagine there's a regularization penalty to post length so feedback that would lengthen the post should be carefully considered.

Post title: {title}

Post content:
{post_text}

Respond with valid JSON containing your feedback:
{{"feedback": "<your 1-3 key points of actionable feedback that would improve this post>"}}
"""


PROMPT_STEP2_POST_EVALUATE = """You are evaluating feedback on an EA forum post. Your task is to 
critically rate the feedback. You should be assessing to what extent
the feedback identifies what amount to major mistakes made by the author. To what extent does the feedback
identify 'own goals' that the author would be embarrassed to have made if they're brought to the author's attention.

Keep in mind also that reader time is precious - imagine there's a regularization penalty to post length so feedback that would lengthen the post should be carefully considered.

Post title: {title}

Post content:
{post_text}

Proposed feedback:
{feedback}

Rate on a 1-10 scale how useful the feedback is for improving this post. We have very high standards for this metric:
- 1: Feedback is irrelevant or would make the post worse
- 3: Moderate improvements, addressing the feedback would fix moderate errors in the post. Or large improvements however addressing the feedback would require lengthening the post significantly. Or the feedback is relevant but the post is a link post with supporting text intentionally elsewhere.
- 5: Large improvements, addresses key mistakes made by the author. And addressing them would not significantly lengthen the post.
- 7: Critical improvements, without addressing the feedback, the post is much worse than it otherwise would be. There are major mistakes made by the author.
- 10: Critical improvements, without addressing the feedback, the main thesis is clearly wrong. The author will likely be hugely embarrassed when they realize they've made these mistakes.

Respond with valid JSON:
{{"feedback_score": <int 1-10>, "improvement_potential": "<brief explanation of why this score>"}}
"""


def compute_post_robustness(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> PostRobustness:
    """Compute robustness score for a post using two-step evaluation.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        PostRobustness metric object
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
    return PostRobustness(
        post_id=post.post_id,
        robustness_score=11 - int(result["feedback_score"]),
        actionable_feedback=feedback,
        improvement_potential=result["improvement_potential"]
    )