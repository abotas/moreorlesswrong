from typing import Literal
from pydantic import BaseModel
import json

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair


class ReasoningQualityV2(BaseModel):
    post_id: str
    reasoning_quality_score: int  # 1-10 reasoning quality score
    thesis: str  # The identified main thesis
    logical_arguments: str  # Key logical arguments identified
    explanation: str  # How effectively arguments support thesis
    
    @classmethod
    def metric_name(cls) -> str:
        return "ReasoningQualityV2"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["reasoning_quality_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "reasoning_quality_score": "Reasoning Quality Score"
        }


PROMPT_REASONING_QUALITY = """Evaluate the quality of reasoning in this EA Forum post.

First identify the thesis of the post. Then identify the logical arguments made to support the thesis. Finally assess how effective the arguments are at supporting the thesis.

Once you've identified the thesis and arguments, consider:
- How logically sound and well-structured are the arguments?
- Are the logical connections between premises and conclusions valid?
- Does the reasoning flow coherently from point to point?
- Are there any logical fallacies or reasoning errors?
- Are there plausible counter-arguments and are these adequately addressed?

Post content to grade:
{title}
{post_text}

Step 1: What is the main thesis or central argument of this post? (Be specific and concise)

Step 2: What logical arguments does the post make? Consider:
- Deductive reasoning and logical inferences
- Analogies and comparative reasoning
- Causal arguments and mechanism explanations
- Theoretical frameworks and conceptual analysis

Step 3: How effectively does the reasoning support the thesis?

Rubric:
Grade on a 1-10 scale for OPTIMAL reasoning quality:
- 1-2: No coherent reasoning OR deeply flawed logical structure
- 3-4: Weak reasoning - significant logical gaps, fallacies, or unsound arguments
- 5-6: Moderate reasoning - some solid arguments but notable weaknesses or gaps
- 7-8: Strong reasoning - well-structured arguments that effectively support the thesis
- 9-10: Optimally reasoned - rigorous logic with just the right level of argumentation, neither under-argued nor redundantly over-argued

Consider:
- Logical soundness: Are the inferences valid?
- Sufficiency: Is there enough reasoning to support the conclusion?
- Efficiency: Is the reasoning appropriately concise vs redundant?
- Structure: Does the logic flow coherently?
- Rigor: How analytically robust is the thinking?

Respond with JSON:
{{
    "thesis": "<main thesis/argument of the post>",
    "logical_arguments": "<summary of key logical arguments made>",
    "reasoning_quality_score": <int 1-10>,
    "explanation": "<explanation of how effectively the reasoning supports the thesis>"
}}
"""


def compute_reasoning_quality_v2(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> ReasoningQualityV2:
    post_text = post.markdown_content or post.html_body or ""
    
    prompt = PROMPT_REASONING_QUALITY.format(
        title=post.title,
        post_text=post_text
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    result = parse_json_with_repair(raw_content)
    
    return ReasoningQualityV2(
        post_id=post.post_id,
        reasoning_quality_score=result["reasoning_quality_score"],
        thesis=result["thesis"],
        logical_arguments=result["logical_arguments"],
        explanation=result["explanation"]
    )