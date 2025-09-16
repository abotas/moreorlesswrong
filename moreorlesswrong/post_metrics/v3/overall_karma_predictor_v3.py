"""Overall karma predictor synthesis metric that combines epistemic and engagement metrics."""

from typing import Literal, List
from pydantic import BaseModel

from llm_client import client
from json_utils import parse_json_with_repair
from models import Post
from db import get_n_most_recent_posts_in_same_cluster
from metric_protocol import Metric, MetricContext

# Import the V3 metric types and evaluation criteria we'll be synthesizing
from post_metrics.v3.value_v3 import ValueV3, VALUE_EVALUATION_CRITERIA
from post_metrics.v3.reasoning_quality_v3 import ReasoningQualityV3, REASONING_QUALITY_EVALUATION_CRITERIA
from post_metrics.v3.cooperativeness_v3 import CooperativenessV3, COOPERATIVENESS_EVALUATION_CRITERIA
from post_metrics.v3.precision_v3 import PrecisionV3, PRECISION_EVALUATION_CRITERIA
from post_metrics.v3.empirical_evidence_quality_v3 import EmpiricalEvidenceQualityV3, EMPIRICAL_EVIDENCE_QUALITY_EVALUATION_CRITERIA
from post_metrics.v3.memetic_potential_v3 import MemeticPotentialV3, MEMETIC_POTENTIAL_EVALUATION_CRITERIA
from post_metrics.v3.author_aura_v3 import AuthorAuraV3
from post_metrics.v3.controversy_temperature_v3 import ControversyTemperatureV3, CONTROVERSY_TEMPERATURE_EVALUATION_CRITERIA


class OverallKarmaPredictorV3(Metric):
    """Synthesis metric that predicts karma based on epistemic and engagement metrics."""
    
    post_id: str
    predicted_karma_score: int  # Predicted karma potential
    karma_analysis: str  # Analysis of karma prediction factors
    epistemic_factors: str  # How epistemic quality affects karma
    engagement_factors: str  # How engagement factors affect karma
    related_posts_context: str  # Context from related posts and their karma
    
    @classmethod
    def dependencies(cls) -> List[str]:
        """Return the V3 metrics required for karma prediction."""
        return [
            "ValueV3",
            "ReasoningQualityV3",
            "CooperativenessV3",
            "PrecisionV3",
            "EmpiricalEvidenceQualityV3",
            "MemeticPotentialV3",
            "AuthorAuraV3",
            "ControversyTemperatureV3"
        ]
    
    @classmethod
    def metric_name(cls) -> str:
        return "OverallKarmaPredictorV3"
    
    @classmethod
    def metric_score_fields(cls) -> List[str]:
        return ["predicted_karma_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "predicted_karma_score": "Predicted Karma Score"
        }

    @classmethod
    def compute(cls, post: Post, context: MetricContext, **metrics) -> "OverallKarmaPredictorV3":
        """Predict karma based on epistemic and engagement metrics with context from related posts.

        Args:
            post: The post to evaluate
            context: Shared metric computation context
            **metrics: Dictionary containing computed dependency metrics

        Returns:
            OverallKarmaPredictorV3 metric object
        """
        # Extract dependency metrics
        value_v3 = metrics["ValueV3"]
        reasoning_quality_v3 = metrics["ReasoningQualityV3"]
        cooperativeness_v3 = metrics["CooperativenessV3"]
        precision_v3 = metrics["PrecisionV3"]
        empirical_evidence_quality_v3 = metrics["EmpiricalEvidenceQualityV3"]
        memetic_potential_v3 = metrics["MemeticPotentialV3"]
        author_aura_v3 = metrics["AuthorAuraV3"]
        controversy_temperature_v3 = metrics["ControversyTemperatureV3"]

        # Ensure all inputs have the same post_id
        post_ids = {
            value_v3.post_id,
            reasoning_quality_v3.post_id,
            cooperativeness_v3.post_id,
            precision_v3.post_id,
            empirical_evidence_quality_v3.post_id,
            memetic_potential_v3.post_id,
            author_aura_v3.post_id,
            controversy_temperature_v3.post_id
        }

        if len(post_ids) != 1:
            raise ValueError(f"All metrics must be for the same post. Got post_ids: {post_ids}")

        post_id = post_ids.pop()

        # Get post text
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

        # Format related posts karma context with full text previews
        related_posts_info = []
        for i, related_post in enumerate(related_posts, 1):
            karma = related_post.base_score or 0
            title = related_post.title

            # Get first 20k characters of post content
            post_content = related_post.markdown_content or related_post.html_body or ""
            content_preview = post_content[:20_000]
            if len(post_content) > 20_000:
                content_preview += "..."

            author_name = related_post.author_display_name or "Unknown"

            related_post_info = f"""
Related Post #{i}: "{title}" by {author_name} (Karma: {karma})
Content preview:
```
{content_preview}
```
"""
            related_posts_info.append(related_post_info)

        related_posts_context = "\\n".join(related_posts_info) if related_posts_info else "No related posts found for context."

        # Format the prompt with all metric data
        prompt = PROMPT_OVERALL_KARMA_PREDICTOR_V3.format(
            # Post content
            title=post.title,
            post_text=post_text,
            related_posts_context=related_posts_context,

            # Epistemic Quality Metrics
            value_score=value_v3.value_score,
            value_criteria=VALUE_EVALUATION_CRITERIA,
            value_analysis=value_v3.analysis,

            reasoning_score=reasoning_quality_v3.reasoning_quality_score,
            reasoning_criteria=REASONING_QUALITY_EVALUATION_CRITERIA,
            reasoning_arguments=reasoning_quality_v3.logical_arguments,
            reasoning_explanation=reasoning_quality_v3.explanation,

            cooperativeness_score=cooperativeness_v3.cooperativeness_score,
            cooperativeness_criteria=COOPERATIVENESS_EVALUATION_CRITERIA,
            cooperativeness_analysis=cooperativeness_v3.analysis_of_cooperativeness,

            precision_score=precision_v3.precision_score,
            precision_criteria=PRECISION_EVALUATION_CRITERIA,
            precision_analysis=precision_v3.analysis,

            empirical_score=empirical_evidence_quality_v3.empirical_evidence_quality_score,
            empirical_criteria=EMPIRICAL_EVIDENCE_QUALITY_EVALUATION_CRITERIA,
            empirical_claims=empirical_evidence_quality_v3.empirical_claims,
            empirical_analysis=empirical_evidence_quality_v3.analysis,

            # Engagement Factors
            memetic_score=memetic_potential_v3.memetic_potential_score,
            memetic_criteria=MEMETIC_POTENTIAL_EVALUATION_CRITERIA,
            memetic_analysis=memetic_potential_v3.analysis,

            author_score=author_aura_v3.ea_fame_score,
            author_analysis=author_aura_v3.analysis,

            controversy_score=controversy_temperature_v3.controversy_temperature_score,
            controversy_criteria=CONTROVERSY_TEMPERATURE_EVALUATION_CRITERIA,
            controversy_thesis=controversy_temperature_v3.identification_of_thesis_and_main_arguments,
            controversy_discussion=controversy_temperature_v3.discussion_and_analysis_of_controversy
        )

        response = client.chat.completions.create(
            model=context.model,
            messages=[{"role": "user", "content": prompt}]
        )

        raw_content = response.choices[0].message.content
        result = parse_json_with_repair(raw_content)

        return cls(
            post_id=post_id,
            predicted_karma_score=result["predicted_karma_score"],
            karma_analysis=result["karma_analysis"],
            epistemic_factors=result["epistemic_factors"],
            engagement_factors=result["engagement_factors"],
            related_posts_context=result["related_posts_context"]
        )


PROMPT_OVERALL_KARMA_PREDICTOR_V3 = """Predict the karma potential for this EA Forum post based on both epistemic quality and engagement factors.

First, here is the post being evaluated:

Title: {title}

Post content:
```
{post_text}
```

For context, here are recent related posts from the same topic cluster and their actual karma scores:
{related_posts_context}

You are provided with 8 metrics for this EA Forum post to help predict karma, along with the evaluation criteria each metric evaluator used:

**EPISTEMIC QUALITY METRICS:**

1. **Value ({value_score}/10)**:
   
   *Evaluation Criteria:*
   ```{value_criteria}```
   
   *Assessment:* 
   ```{value_analysis}```

2. **Reasoning Quality ({reasoning_score}/10)**:
   
   *Evaluation Criteria:*
   ```{reasoning_criteria}```
   
   *Assessment:*
   - Arguments: ```{reasoning_arguments}```
   - Analysis: ```{reasoning_explanation}```

3. **Cooperativeness ({cooperativeness_score}/10)**:
   
   *Evaluation Criteria:*
   ```{cooperativeness_criteria}```
   
   *Assessment:* ```{cooperativeness_analysis}```

4. **Precision ({precision_score}/10)**:
   
   *Evaluation Criteria:*
   ```{precision_criteria}```
   
   *Assessment:* ```{precision_analysis}```

5. **Empirical Evidence Quality ({empirical_score}/10)**:
   
   *Evaluation Criteria:*
   ```{empirical_criteria}```
   
   *Assessment:*
   - Claims: ```{empirical_claims}```
   - Analysis: ```{empirical_analysis}```

**ENGAGEMENT FACTORS:**

6. **Memetic Potential ({memetic_score}/10)**:
   
   *Evaluation Criteria:*
   ```{memetic_criteria}```
   
   *Assessment:* ```{memetic_analysis}```

7. **Controversy Temperature ({controversy_score}/10)**:
   
   *Evaluation Criteria:*
   ```{controversy_criteria}```
   
   *Assessment:*
   - Thesis Analysis: ```{controversy_thesis}```
   - Controversy Discussion: ```{controversy_discussion}```

**And finally:**

8. **Author Fame ({author_score}/10)**:
   
   *Assessment:* ```{author_analysis}```

Author fame encodes both a certain amount of epistemic quality and engagement factors. EA notoriety likely
comes with a track record of quality epistemics, but it's also a signal the author is able to drive engagement and there's some self-fulfilling prophecy
where if you're EA-famous you're more likely to get clicks. 

Your task is to predict the actual karma score of this post based on:
- How epistemic quality factors correlate with karma in EA Forum
- How engagement factors (memetic potential, author reputation, controversy) drive upvotes
- Context from similar posts and their actual karma performance
- The specific content and style of this post

Consider that karma reflects both:
- **Quality signals**: Value, reasoning, cooperativeness (epistemic virtues)
- **Engagement signals**: Memorable insights, author reputation, attention-getting controversy

Respond with JSON:
```json{{
    "related_posts_context": "<brief summary of how the related posts' karma relates to this post's potential>",
    "epistemic_factors": "<how the epistemic quality metrics suggest karma performance>",
    "engagement_factors": "<how author aura, memetic potential, and controversy temperature will drive engagement>", 
    "karma_analysis": "<overall analysis synthesizing all factors for karma prediction>",
    "predicted_karma_score": <int>
}}
```
"""


def compute_overall_karma_predictor_v3(
    post: Post,
    value_v3: ValueV3,
    reasoning_quality_v3: ReasoningQualityV3,
    cooperativeness_v3: CooperativenessV3,
    precision_v3: PrecisionV3,
    empirical_evidence_quality_v3: EmpiricalEvidenceQualityV3,
    memetic_potential_v3: MemeticPotentialV3,
    author_aura_v3: AuthorAuraV3,
    controversy_temperature_v3: ControversyTemperatureV3,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini",
    n_related_posts: int = 5
) -> OverallKarmaPredictorV3:
    """Predict karma based on epistemic and engagement metrics with context from related posts.
    
    This synthesis metric combines both epistemic quality metrics (value, reasoning, etc.)
    and engagement factors (memetic potential, author aura, controversy) along with karma data from
    similar posts to predict the karma potential of the current post.
    
    Args:
        post: The post to evaluate
        value_v3: Computed ValueV3 metric
        reasoning_quality_v3: Computed ReasoningQualityV3 metric  
        cooperativeness_v3: Computed CooperativenessV3 metric
        precision_v3: Computed PrecisionV3 metric
        empirical_evidence_quality_v3: Computed EmpiricalEvidenceQualityV3 metric
        memetic_potential_v3: Computed MemeticPotentialV3 metric
        author_aura_v3: Computed AuthorAuraV3 metric
        controversy_temperature_v3: Computed ControversyTemperatureV3 metric
        model: The model to use for synthesis
        n_related_posts: Number of related posts to use for context
        
    Returns:
        OverallKarmaPredictorV3 metric object
    """
    # Ensure all inputs have the same post_id
    post_ids = {
        value_v3.post_id,
        reasoning_quality_v3.post_id,
        cooperativeness_v3.post_id,
        precision_v3.post_id,
        empirical_evidence_quality_v3.post_id,
        memetic_potential_v3.post_id,
        author_aura_v3.post_id,
        controversy_temperature_v3.post_id
    }
    
    if len(post_ids) != 1:
        raise ValueError(f"All metrics must be for the same post. Got post_ids: {post_ids}")
    
    post_id = post_ids.pop()
    
    # Get post text
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
    
    # Format related posts karma context with full text previews
    related_posts_info = []
    for i, related_post in enumerate(related_posts, 1):
        karma = related_post.base_score or 0
        title = related_post.title
        
        # Get first 20k characters of post content
        post_content = related_post.markdown_content or related_post.html_body or ""
        content_preview = post_content[:20_000]
        if len(post_content) > 20_000:
            content_preview += "..."
        
        author_name = related_post.author_display_name or "Unknown"
        
        related_post_info = f"""
Related Post #{i}: "{title}" by {author_name} (Karma: {karma})
Content preview:
```
{content_preview}
```
"""
        related_posts_info.append(related_post_info)
    
    related_posts_context = "\n".join(related_posts_info) if related_posts_info else "No related posts found for context."
    
    # Format the prompt with all metric data
    prompt = PROMPT_OVERALL_KARMA_PREDICTOR_V3.format(
        # Post content
        title=post.title,
        post_text=post_text,
        related_posts_context=related_posts_context,
        
        # Epistemic Quality Metrics
        value_score=value_v3.value_score,
        value_criteria=VALUE_EVALUATION_CRITERIA,
        value_analysis=value_v3.analysis,
        
        reasoning_score=reasoning_quality_v3.reasoning_quality_score,
        reasoning_criteria=REASONING_QUALITY_EVALUATION_CRITERIA,
        reasoning_arguments=reasoning_quality_v3.logical_arguments,
        reasoning_explanation=reasoning_quality_v3.explanation,
        
        cooperativeness_score=cooperativeness_v3.cooperativeness_score,
        cooperativeness_criteria=COOPERATIVENESS_EVALUATION_CRITERIA,
        cooperativeness_analysis=cooperativeness_v3.analysis_of_cooperativeness,
        
        precision_score=precision_v3.precision_score,
        precision_criteria=PRECISION_EVALUATION_CRITERIA,
        precision_analysis=precision_v3.analysis,
        
        empirical_score=empirical_evidence_quality_v3.empirical_evidence_quality_score,
        empirical_criteria=EMPIRICAL_EVIDENCE_QUALITY_EVALUATION_CRITERIA,
        empirical_claims=empirical_evidence_quality_v3.empirical_claims,
        empirical_analysis=empirical_evidence_quality_v3.analysis,
        
        # Engagement Factors
        memetic_score=memetic_potential_v3.memetic_potential_score,
        memetic_criteria=MEMETIC_POTENTIAL_EVALUATION_CRITERIA,
        memetic_analysis=memetic_potential_v3.analysis,
        
        author_score=author_aura_v3.ea_fame_score,
        author_analysis=author_aura_v3.analysis,
        
        controversy_score=controversy_temperature_v3.controversy_temperature_score,
        controversy_criteria=CONTROVERSY_TEMPERATURE_EVALUATION_CRITERIA,
        controversy_thesis=controversy_temperature_v3.identification_of_thesis_and_main_arguments,
        controversy_discussion=controversy_temperature_v3.discussion_and_analysis_of_controversy
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    result = parse_json_with_repair(raw_content)
    
    return OverallKarmaPredictorV3(
        post_id=post_id,
        predicted_karma_score=result["predicted_karma_score"],
        karma_analysis=result["karma_analysis"],
        epistemic_factors=result["epistemic_factors"],
        engagement_factors=result["engagement_factors"],
        related_posts_context=result["related_posts_context"]
    )