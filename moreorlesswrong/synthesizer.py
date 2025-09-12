from typing import List, Literal
from pydantic import BaseModel
import json

from models import Post
from llm_client import client
from json_utils import parse_json_with_repair


SYNTHESIZER_PROMPT = """You're an information synthesis expert working to find pertinent information from previous EA Forum posts that might be useful in evaluating a new EA Forum post.

We are planning on evaluating a new post for this metric: {metric_name}, and first need to synthesize useful context from previous posts.

This is how we'll evaluate {metric_name}:
{prompt_for_metric_generation_substr}

Your task is to consider the new post along side a set of previous posts and synthesize any useful context from the previous posts that'd help in evaluating our new post for {metric_name}.

The new post we'll later evaluate:
```
{new_post_title}
{new_post_preview}
```

Previous posts from which to extract any useful context that would help in evaluating for {metric_name}:
```
{previous_posts_str}
```

- First consider the posts and the metric we're going to evaluate the new post for. 
- Second, reason about what kinds of summaries or excerpts from the previous posts would be helpful context when evaluating the new post for {metric_name}.
- Third, extract the most useful / relevant excerpts from the previous posts for evaluating {metric_name}.
- Finally, synthesize this into the context_for_downstream_evaluator_model json.

You may write out your thoughts before the providing the final JSON response.

Return a JSON response:
{{
    "reasoning_about_what_kinds_of_context_would_be_useful": "<reasoning about what kinds of context would be useful for evaluating the new post for {metric_name}>",
    "relevant_excerpts_or_paraphrased_summaries": "<relevant excerpts or paraphrased summaries from previous posts that might help in evaluating the new post for {metric_name}>",
    "context_for_downstream_evaluator_model": "<synthesized context. This should be a focused few paragraphs containing the most useful context from previous posts that might help in evaluating the new post for {metric_name}. This should not simply be a summary of previous posts. It should in a focused manner cite and review any information or claims from the previous posts that might be helpful in evaluating the new post for {metric_name}.>"
}}
If there are no related previous posts or no useful relevant context is found return something like this
{{
    "context_for_downstream_evaluator_model": "No relevant context found"
}}
"""


def format_previous_posts(posts: List[Post]) -> str:
    """Format previous posts for inclusion in the synthesis prompt."""
    if not posts:
        return "No previous posts available"
    
    formatted = []
    for i, post in enumerate(posts, 1):
        content = post.markdown_content or post.html_body or "No content available"
        
        formatted.append(f"""
Post {i}:
- Title: {post.title}
- Author: {post.author_display_name or 'Unavailable'}
- Karma: {post.base_score or 'Unavailable'}
- Full content:
{content}
""")
    
    return "\n".join(formatted)


def synthesize_context(
    new_post: Post,
    previous_posts: List[Post],
    metric_name: str,
    metric_evaluation_prompt: str,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> str:
    """Synthesize useful context from previous posts for evaluating a new post.
    
    Args:
        new_post: The post to be evaluated
        previous_posts: List of related previous posts
        metric_name: Name of the metric being evaluated
        metric_evaluation_prompt: The evaluation criteria/prompt for the metric
        model: The model to use for synthesis
        
    Returns:
        Synthesized context string
    """
    # print(f'FOUND *{len(previous_posts)}* RELATED PREV POSTS')

    if len(previous_posts) == 0:
        return "Unfortunately no previous posts are available. Please go without."
    new_post_content = new_post.markdown_content or new_post.html_body or "No content available"
    
    # Format previous posts
    previous_posts_str = format_previous_posts(previous_posts)

    # Create synthesis prompt
    prompt = SYNTHESIZER_PROMPT.format(
        metric_name=metric_name,
        prompt_for_metric_generation_substr=metric_evaluation_prompt,
        new_post_title=new_post.title,
        new_post_preview=new_post_content,
        previous_posts_str=previous_posts_str
    )
    
    # Get synthesis from LLM
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw_content = response.choices[0].message.content
    result = parse_json_with_repair(raw_content)

    # print('--------------------------------Prompt to synthesizer:--------------------------------')
    # print(prompt[:1000])
    # print('\n\n')
    # print(raw_content)
    # print('----------------------------------------------------------------')
    
    return result["context_for_downstream_evaluator_model"]