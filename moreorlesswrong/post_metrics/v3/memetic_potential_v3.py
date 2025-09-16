from typing import Literal
from pydantic import BaseModel

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair
from db import get_n_most_recent_posts_in_same_cluster
from prev_post_synthesizer import synthesize_context
from raw_context_formatter import format_raw_related_posts
from metric_protocol import Metric, MetricContext


class MemeticPotentialV3(Metric):
    post_id: str
    memetic_potential_score: int  # 1-10 memetic potential score
    analysis: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "MemeticPotentialV3"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["memetic_potential_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "memetic_potential_score": "Memetic Potential Score"
        }

    @classmethod
    def compute(cls, post: Post, context: MetricContext) -> "MemeticPotentialV3":
        """Compute memetic potential score for a post with synthesized information from related posts.

        Args:
            post: The post to evaluate
            context: Shared metric computation context

        Returns:
            MemeticPotentialV3 metric object
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
            metric_name="Memetic Potential",
            metric_evaluation_prompt=MEMETIC_POTENTIAL_EVALUATION_CRITERIA,
            model=context.model,
            synthesizer_focus_area=SYNTHESIZER_FOCUS_AREA
        )

        prompt = PROMPT_MEMETIC_POTENTIAL_V3.format(
            evaluation_criteria=MEMETIC_POTENTIAL_EVALUATION_CRITERIA,
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
            memetic_potential_score=result["memetic_potential_score"],
            analysis=result["analysis"]
        )


# Extract the evaluation criteria for the synthesizer
MEMETIC_POTENTIAL_EVALUATION_CRITERIA = """Evaluate the MEMETIC POTENTIAL of this EA Forum post.

Memetic potential refers to how likely this post is to:
- Create new terminology that others will adopt
- Introduce frameworks or schemas that become reference points
- Generate concepts that spread through the EA community
- Produce ideas that get cited and built upon

Rubric:
Grade on a 1-10 scale for memetic potential.

High memetic potential (7-10):
- Contains compelling new terms or acronyms or quotable phrases that map to important and compelling concepts
- Presents concise novel frameworks for thinking about problems
- Creates pithy models others will reference
- Introduces concepts that simplify complex ideas

Medium memetic potential (4-6):
- Is likely missing something for high memetic potential, i.e. importance, relevance, or clarity
- e.g. presents important concepts but without clear concise or pithy framing
- or filled with simple clear ideas but without appropriate novelty or importance

Low memetic potential (1-3):
- Discusses existing ideas without new framing
- Commentary on established concepts
- Personal reflections without generalizable insights
- Technical details without broader applicability

Consider:
- Does this introduce new vocabulary or concepts?
- Will other posts likely reference this framework?
- Are there quotable insights or principles?
- Does it create tools for thinking about problems?
- Will this change how EAs discuss this topic?
"""

SYNTHESIZER_FOCUS_AREA = """Look for:
- Memorable elements, frameworks, or terminology that previous posts introduced
- Whether this posts memorable elements are duplications of existing memes and frameworks from previous posts
- Whether this post's ideas build on or compete or resolve conflicts from previous related posts"""


PROMPT_MEMETIC_POTENTIAL_V3 = """{evaluation_criteria}

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
    "analysis": "<Analysis of memetic potential>"
    "memetic_potential_score": <int 1-10>,
}}
```
"""


def compute_memetic_potential_v3(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini",
    bypass_synthesizer: bool = False,
    n_related_posts: int = 5,
) -> MemeticPotentialV3:
    """Compute memetic potential score for a post with synthesized information from related posts.
    
    Args:
        post: The post to evaluate
        model: The model to use for evaluation
        bypass_synthesizer: If True, use raw related posts instead of synthesized context
        
    Returns:
        MemeticPotentialV3 metric object
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
            metric_name="Memetic Potential",
            metric_evaluation_prompt=MEMETIC_POTENTIAL_EVALUATION_CRITERIA,
            model=model,
            synthesizer_focus_area=SYNTHESIZER_FOCUS_AREA
        )

    prompt = PROMPT_MEMETIC_POTENTIAL_V3.format(
        evaluation_criteria=MEMETIC_POTENTIAL_EVALUATION_CRITERIA,
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
    
    return MemeticPotentialV3(
        post_id=post.post_id,
        memetic_potential_score=result["memetic_potential_score"],
        analysis=result["analysis"]
    )