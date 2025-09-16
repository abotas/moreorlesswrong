from typing import Literal
from pydantic import BaseModel

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair
from metric_protocol import Metric, MetricContext


class GptFullOBOQualityV0(Metric):
    post_id: str
    quality_score: int  # 1-10 overall quality score
    explanation: str

    @classmethod
    def metric_name(cls) -> str:
        return "GptFullOBOQualityV0"

    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["quality_score"]

    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "quality_score": "GPT-Full OBO Quality Score"
        }

    @classmethod
    def compute(cls, post: Post, context: MetricContext) -> "GptFullOBOQualityV0":
        """Compute GPT-Full OBO quality score for a post.

        Args:
            post: The post to evaluate
            context: Metric computation context (ignored for V0 metrics)

        Returns:
            GptFullOBOQualityV0 metric object
        """
        prompt = PROMPT_GPT_OBO_QUALITY.format(
            title=post.title,
            content=post.markdown_content
        )

        response = client.chat.completions.create(
            model="gpt-5",  # Hardcoded to gpt-5
            messages=[{"role": "user", "content": prompt}]
        )

        raw_content = response.choices[0].message.content
        result = parse_json_with_repair(raw_content)

        return cls(
            post_id=post.post_id,
            quality_score=result["quality_score"],
            explanation=result["explanation"]
        )


PROMPT_GPT_OBO_QUALITY = """Consider the overall quality of this EA forum post. Rate on a scale 1-10.
```
{title}
{content}
```
Return your response in this exact JSON format:
{{
    "explanation": "Brief explanation of why you gave this rating",
    "quality_score": <score from 1-10>
}}
"""