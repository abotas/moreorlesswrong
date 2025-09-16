from typing import Literal
from pydantic import BaseModel

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair
from metric_protocol import Metric, MetricContext


class GptNanoOBOEpistemicQualityV0(Metric):
    post_id: str
    epistemic_quality_score: int  # 1-10 overall epistemic quality score
    explanation: str

    @classmethod
    def metric_name(cls) -> str:
        return "GptNanoOBOEpistemicQualityV0"

    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["epistemic_quality_score"]

    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "epistemic_quality_score": "GPT-Nano OBO Epistemic Quality Score"
        }

    @classmethod
    def compute(cls, post: Post, context: MetricContext) -> "GptNanoOBOEpistemicQualityV0":
        """Compute GPT-Nano OBO epistemic quality score for a post.

        Args:
            post: The post to evaluate
            context: Metric computation context (ignored for V0 metrics)

        Returns:
            GptNanoOBOEpistemicQualityV0 metric object
        """
        prompt = PROMPT_GPT_OBO_EPISTEMIC.format(
            title=post.title,
            content=post.markdown_content
        )

        response = client.chat.completions.create(
            model="gpt-5-nano",  # Hardcoded to gpt-5-nano
            messages=[{"role": "user", "content": prompt}]
        )

        raw_content = response.choices[0].message.content
        result = parse_json_with_repair(raw_content)

        return cls(
            post_id=post.post_id,
            epistemic_quality_score=result["epistemic_quality_score"],
            explanation=result["explanation"]
        )


PROMPT_GPT_OBO_EPISTEMIC = """Consider the holistic epistemic quality of this EA forum post. Rate on a scale 1-10.
```
{title}
{content}
```
Return your response in this exact JSON format:
{{
    "explanation": "Brief explanation of why you gave this rating",
    "epistemic_quality_score": <score from 1-10>
}}
"""