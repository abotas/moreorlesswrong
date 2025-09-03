from typing import Literal
from pydantic import BaseModel

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair


class ControversyTemperatureV2(BaseModel):
    post_id: str
    controversy_temperature_score: int  # 1-10 controversy temperature score
    explanation: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "ControversyTemperatureV2"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["controversy_temperature_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "controversy_temperature_score": "Controversy Temperature Score"
        }


PROMPT_CONTROVERSY_TEMPERATURE = """Evaluate the CONTROVERSY TEMPERATURE of this EA Forum post.

Controversy temperature measures whether the post hits the "goldilocks zone" of productive disagreement - controversial enough to generate discussion, but not so controversial that it becomes unproductive flamebait.

Rubric:
Grade on a 1-10 scale for controversy temperature.

OPTIMAL controversy temperature (7-10 points):
- Thoughtfully challenges EA orthodoxy with strong arguments
- Presents minority views with good faith reasoning
- Questions sacred cows constructively
- Makes claims that reasonable EAs would disagree about
- Invites productive debate on important uncertainties
- Examples: "Against longtermism" (with good arguments), "EA is too focused on AI", "We should deprioritize animal welfare"

MODERATE controversy temperature (4-6 points):
- Discusses known areas of disagreement neutrally
- Presents multiple perspectives on debated topics
- Makes mildly controversial empirical claims
- Suggests minor course corrections
- Examples: "Which cause area deserves more funding?", "Comparing career paths"

SUBOPTIMAL controversy temperature (1-3 points):
Either TOO COLD (boring consensus):
- States obvious truths everyone agrees with
- Discusses settled questions
- Makes uncontroversial claims
- Examples: "Malaria is bad", "We should be effective"

Or TOO HOT (unproductive flamebait):
- Ad hominem attacks
- Bad faith arguments
- Inflammatory rhetoric without substance
- Tribalistic us-vs-them framing
- Examples: "EA is a cult", "All EAs are wrong about everything"

Post content to grade:
{title}

{post_text}

Consider:
- Does this challenge important assumptions productively?
- Will it generate thoughtful disagreement or just agreement/anger?
- Is controversy backed by good faith reasoning?
- Does it avoid both boring consensus AND destructive flame wars?
- Will this create productive discussion?

Respond with JSON:
{{
    "controversy_temperature_score": <int 1-10>,
    "explanation": "<brief explanation of controversy level and whether it's productive>"
}}
"""


def compute_controversy_temperature_v2(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> ControversyTemperatureV2:
    """Compute controversy temperature score for a post.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        
    Returns:
        ControversyTemperatureV2 metric object
    """
    post_text = post.markdown_content or post.html_body or ""
    
    prompt = PROMPT_CONTROVERSY_TEMPERATURE.format(
        title=post.title,
        post_text=post_text
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    result = parse_json_with_repair(raw_content)
    
    return ControversyTemperatureV2(
        post_id=post.post_id,
        controversy_temperature_score=result["controversy_temperature_score"],
        explanation=result["explanation"]
    )