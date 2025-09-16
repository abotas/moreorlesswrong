from typing import Literal
from pydantic import BaseModel

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair
from db import get_n_most_recent_posts_in_same_cluster
from prev_post_synthesizer import synthesize_context
from raw_context_formatter import format_raw_related_posts
from metric_protocol import Metric, MetricContext


class ControversyTemperatureV3(Metric):
    post_id: str
    controversy_temperature_score: int  # 1-10 controversy temperature score
    identification_of_thesis_and_main_arguments: str
    discussion_and_analysis_of_controversy: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "ControversyTemperatureV3"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["controversy_temperature_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "controversy_temperature_score": "Controversy Temperature Score"
        }

    @classmethod
    def compute(cls, post: Post, context: MetricContext) -> "ControversyTemperatureV3":
        """Compute controversy temperature score for a post with synthesized information from related posts.

        Args:
            post: The post to evaluate
            context: Shared metric computation context

        Returns:
            ControversyTemperatureV3 metric object
        """
        post_text = post.markdown_content or post.html_body or ""

        related_posts = get_n_most_recent_posts_in_same_cluster(
            post_id=post.post_id,
            cluster_cardinality=12,
            n=context.n_related_posts
        )
        if len(related_posts) < context.n_related_posts:
            related_posts2 = get_n_most_recent_posts_in_same_cluster(
                post_id=post.post_id,
                cluster_cardinality=5,
                n=context.n_related_posts*2
            )
            postids = [p.post_id for p in related_posts]
            related_posts = related_posts + [p for p in related_posts2 if p.post_id not in postids]
            related_posts = related_posts[:context.n_related_posts]

        # Always use synthesizer approach (removed bypass_synthesizer)
        synthesized_info = synthesize_context(
            new_post=post,
            previous_posts=related_posts,
            metric_name="Controversy Temperature",
            metric_evaluation_prompt=CONTROVERSY_TEMPERATURE_EVALUATION_CRITERIA,
            model=context.model,
            synthesizer_focus_area=SYNTHESIZER_FOCUS_AREA
        )

        prompt = PROMPT_CONTROVERSY_TEMPERATURE_V3.format(
            evaluation_criteria=CONTROVERSY_TEMPERATURE_EVALUATION_CRITERIA,
            title=post.title,
            post_text=post_text,
            synthesized_info=synthesized_info
        )

        response = client.chat.completions.create(
            model=context.model,
            messages=[{"role": "user", "content": prompt}]
        )

        raw_content = response.choices[0].message.content
        result = parse_json_with_repair(raw_content)

        return cls(
            post_id=post.post_id,
            controversy_temperature_score=result["controversy_temperature_score"],
            identification_of_thesis_and_main_arguments=result["identification_of_thesis_and_main_arguments"],
            discussion_and_analysis_of_controversy=result["discussion_and_analysis_of_controversy"]
        )


# Extract the evaluation criteria for the synthesizer
CONTROVERSY_TEMPERATURE_EVALUATION_CRITERIA = """Evaluate the CONTROVERSY TEMPERATURE of this EA Forum post.

Controversy temperature measures how likely a post is to capture attention and generate engagement through its controversial stance.
Consider:
- Will this capture attention and generate clicks/engagement?
- Is it controversial enough to be interesting but credible enough to be taken seriously?
- Does it hit the sweet spot between boring consensus and absurd extremism?

First you'll identify the main arguments from the post and which ones might be controversial. 
Then you'll discuss whether they are likely to be controversial -- do they challenge beliefs of some EAs? are the arguments thought-
provoking but credible? What is the tone of the post? Given your discussion what's your analysis of the overall controversy of the post?
Finally you'll provide a score on a 1-10 scale.

Rubric:
Grade on a 1-10 scale for attention-getting controversy potential.

HIGHEST ATTENTION POTENTIAL (8-10 points):
Posts in the "goldilocks zone" - controversial enough to demand attention, but credible enough to be taken seriously:
- Thoughtfully challenges EA orthodoxy with strong arguments
- Presents minority views with good faith reasoning
- Questions sacred cows constructively
- Makes claims that reasonable EAs would strongly disagree about
- Invites heated but productive debate on important uncertainties
- Examples: "We should deprioritize {{your least favorite cause area}}" with good arguments

MODERATE ATTENTION POTENTIAL (5-7 points):
Posts with some controversy that will get noticed but not dominate discussion:
- Discusses known areas of disagreement with a clear stance
- Takes a side on debated topics
- Suggests meaningful course corrections
- Examples: "Which cause area deserves more funding?", "Comparing career paths", "EA needs more diversity"

LOWER ATTENTION POTENTIAL (3-4 points):
Posts with mild controversy or familiar debates:
- Rehashes well-worn disagreements
- Makes minor controversial points
- Gentle critiques of mainstream views
- Examples: "EA could be more welcoming", "We should consider X cause area too"

LOWEST ATTENTION POTENTIAL (1-2 points):
Posts unlikely to capture attention due to being either too bland and therefore uninteresting OR too absurd and therefore noncontroversial
because readership all agrees the post is bad:
Excessively bland:
- States obvious truths everyone agrees with
- Rehashes settled questions without novel insights
Too absurd:
- Pure ad hominem attacks without substance
- Conspiracy theories with no evidence
- Obviously trolling or bad faith
- Examples: "Charity is always wrong"
"""

SYNTHESIZER_FOCUS_AREA = """Look for:
- Ways the new post may be controversial that may not be obvious to an evaluator that only reads the new post itself and not the related previous posts
- Whether this post's stance aligns with or challenges positions taken in related previous posts
- Differences in tone or rhetoric between this post and related previous posts
"""


PROMPT_CONTROVERSY_TEMPERATURE_V3 = """{evaluation_criteria}

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
    "identification_of_thesis_and_main_arguments": "<arguments identification>",
    "discussion_and_analysis_of_controversy": "<discussion and analysis>",
    "controversy_temperature_score": <int 1-10>,
}}
```
"""


def compute_controversy_temperature_v3(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini",
    bypass_synthesizer: bool = False,
    n_related_posts: int = 5,
) -> ControversyTemperatureV3:
    """Compute controversy temperature score for a post with synthesized information from related posts.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        bypass_synthesizer: If True, use raw related posts instead of synthesized context
        
    Returns:
        ControversyTemperatureV3 metric object
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
            metric_name="Controversy Temperature",
            metric_evaluation_prompt=CONTROVERSY_TEMPERATURE_EVALUATION_CRITERIA,
            model=model,
            synthesizer_focus_area=SYNTHESIZER_FOCUS_AREA
        )

    prompt = PROMPT_CONTROVERSY_TEMPERATURE_V3.format(
        evaluation_criteria=CONTROVERSY_TEMPERATURE_EVALUATION_CRITERIA,
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
    
    return ControversyTemperatureV3(
        post_id=post.post_id,
        controversy_temperature_score=result["controversy_temperature_score"],
        identification_of_thesis_and_main_arguments=result["identification_of_thesis_and_main_arguments"],
        discussion_and_analysis_of_controversy=result["discussion_and_analysis_of_controversy"]
    )
