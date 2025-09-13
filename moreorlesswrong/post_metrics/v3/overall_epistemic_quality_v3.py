"""Overall epistemic quality synthesis metric that combines multiple V3 epistemic metrics."""

from typing import Literal, List
from pydantic import BaseModel

from synthesis_metric_base import SynthesisMetric
from llm_client import client
from json_utils import parse_json_with_repair
from models import Post

from post_metrics.v3.value_v3 import ValueV3, VALUE_EVALUATION_CRITERIA
from post_metrics.v3.reasoning_quality_v3 import ReasoningQualityV3, REASONING_QUALITY_EVALUATION_CRITERIA
from post_metrics.v3.cooperativeness_v3 import CooperativenessV3, COOPERATIVENESS_EVALUATION_CRITERIA
from post_metrics.v3.precision_v3 import PrecisionV3, PRECISION_EVALUATION_CRITERIA
from post_metrics.v3.empirical_evidence_quality_v3 import EmpiricalEvidenceQualityV3, EMPIRICAL_EVIDENCE_QUALITY_EVALUATION_CRITERIA


class OverallEpistemicQualityV3(SynthesisMetric):
    """Synthesis metric that combines epistemic V3 metrics into overall quality score."""
    
    post_id: str
    overall_epistemic_quality_score: int  # 1-10 overall epistemic quality score
    synthesis_analysis: str  # Analysis of how metrics combine
    key_strengths: str  # Top epistemic strengths
    key_weaknesses: str  # Main epistemic weaknesses
    
    @classmethod
    def required_metrics(cls) -> List[str]:
        """Return the V3 epistemic metrics required for synthesis."""
        return [
            "ValueV3",
            "ReasoningQualityV3",
            "CooperativenessV3",
            "PrecisionV3",
            "EmpiricalEvidenceQualityV3"
        ]
    
    @classmethod
    def metric_name(cls) -> str:
        return "OverallEpistemicQualityV3"
    
    @classmethod
    def metric_score_fields(cls) -> List[str]:
        return ["overall_epistemic_quality_score"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "overall_epistemic_quality_score": "Overall Epistemic Quality Score"
        }


PROMPT_OVERALL_EPISTEMIC_QUALITY_V3 = """
You are an epistemic virtue evaluator of EA Forum posts. 
You are tasked with synthesizing an overall epistemic quality assessment from individual metric evaluators.

Synthesize an overall epistemic quality assessment given analysis from individual metric evaluators.

Here is the post being evaluated:
```
{title}
{post_text}
```

You are provided with 5 epistemic quality metrics for this EA Forum post graded by individual metric evaluators, each scored 1-10, along with the evaluation criteria the metric evaluators used:

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

Your task is to synthesize these into an overall epistemic quality assessment. Value is likely the most central metric. But you need to consider whether the other metrics bolster or invalidate the value of the post.

Consider:
- The Value of the post and if it is bolstered or undermined by other metrics
  - How do these metrics reinforce/contradict each other?
- Are there takeaways that emerge with a top-level view of all the metrics that the focused evaluators might have missed or been
unable to assess?
  - Given the content of the post does one or a subset of the metrics deserve more or less consideration than usual?

Give an overall assessment of the post's aggregate epistemic quality.

Respond with JSON:
```json{{
    "key_strengths": "<1-2 most important epistemic strengths and contributions>",
    "key_weaknesses": "<1-2 main epistemic weaknesses or gaps that hold back the post's overall quality>",
    "overall_epistemic_quality_analysis": "<all things considered analysis of the post's epistemic quality>",
    "overall_epistemic_quality_score": <1-10 overall epistemic quality score>
}}
```
"""


def compute_overall_epistemic_quality_v3(
    post: Post,
    value_v3: ValueV3,
    reasoning_quality_v3: ReasoningQualityV3,
    cooperativeness_v3: CooperativenessV3,
    precision_v3: PrecisionV3,
    empirical_evidence_quality_v3: EmpiricalEvidenceQualityV3,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini",
    n_related_posts: int = 5,
    bypass_synthesizer: bool = False
) -> OverallEpistemicQualityV3:
    # Ensure all inputs have the same post_id
    post_ids = {
        value_v3.post_id,
        reasoning_quality_v3.post_id,
        cooperativeness_v3.post_id,
        precision_v3.post_id,
        empirical_evidence_quality_v3.post_id
    }
    
    if len(post_ids) != 1:
        raise ValueError(f"All metrics must be for the same post. Got post_ids: {post_ids}")
    
    post_id = post_ids.pop()
    
    # Get post text
    post_text = post.markdown_content or post.html_body or ""
    
    # Format the prompt with all metric data
    prompt = PROMPT_OVERALL_EPISTEMIC_QUALITY_V3.format(
        # Post content
        title=post.title,
        post_text=post_text,
        
        # Value
        value_score=value_v3.value_score,
        value_criteria=VALUE_EVALUATION_CRITERIA,
        value_analysis=value_v3.analysis,
        
        # Reasoning Quality
        reasoning_score=reasoning_quality_v3.reasoning_quality_score,
        reasoning_criteria=REASONING_QUALITY_EVALUATION_CRITERIA,
        reasoning_thesis=reasoning_quality_v3.thesis,
        reasoning_arguments=reasoning_quality_v3.logical_arguments,
        reasoning_explanation=reasoning_quality_v3.explanation,
        
        # Cooperativeness
        cooperativeness_score=cooperativeness_v3.cooperativeness_score,
        cooperativeness_criteria=COOPERATIVENESS_EVALUATION_CRITERIA,
        cooperativeness_analysis=cooperativeness_v3.analysis_of_cooperativeness,
        
        # Precision
        precision_score=precision_v3.precision_score,
        precision_criteria=PRECISION_EVALUATION_CRITERIA,
        precision_analysis=precision_v3.analysis,
        
        # Empirical Evidence
        empirical_score=empirical_evidence_quality_v3.empirical_evidence_quality_score,
        empirical_criteria=EMPIRICAL_EVIDENCE_QUALITY_EVALUATION_CRITERIA,
        empirical_thesis=empirical_evidence_quality_v3.thesis,
        empirical_claims=empirical_evidence_quality_v3.empirical_claims,
        empirical_analysis=empirical_evidence_quality_v3.analysis
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    result = parse_json_with_repair(raw_content)
    
    return OverallEpistemicQualityV3(
        post_id=post_id,
        overall_epistemic_quality_score=result["overall_epistemic_quality_score"],
        synthesis_analysis=result["overall_epistemic_quality_analysis"],
        key_strengths=result["key_strengths"],
        key_weaknesses=result["key_weaknesses"]
    )