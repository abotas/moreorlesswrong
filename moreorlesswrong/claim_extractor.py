import json
from typing import List, Literal
from models import Post, Claim
from llm_client import client

PROMPT_CLAIM_EXTRACTION = """You are an expert at parsing text for central claims and arguments. Extract the {n} most CENTRAL and IMPORTANT claims from the provided EA Forum post.

A central claim should:
- Be a substantive assertion or argument made in the text
- Contain all necessary context to understand it independently
    - Include geographic/jurisdictional scope when relevant
    - Include temporal context when relevant  
    - Be specific rather than vague (avoid phrases like 'the proposal' - say which proposal)

Focus on the core arguments and assertions rather than minor supporting details.

Respond with a valid JSON matching this schema, where claim_1 is the most central claim, claim_2 is the second most central, etc:
{{"claim_1": "str", "claim_2": "str", ...}}

If there are fewer than {n} substantive claims, return only what you find.

Here is the EA Forum post to parse:

Title: {title}
Author: {author}

{text}

"""

PROMPT_THESIS_EXTRACTION = """You are an expert at parsing text for central claims and arguments. Extract CENTRAL and IMPORTANT THESIS from the provided EA Forum post.

The thesis should:
- Be a substantive assertion or argument made in the text
- Contain all necessary context to understand it independently
    - Include geographic/jurisdictional scope when relevant
    - Include temporal context when relevant  
    - Be specific rather than vague (avoid phrases like 'the proposal' - say which proposal)

Focus on the core arguments and assertions rather than minor supporting details.

Respond with a valid JSON matching this schema:
{{"claim_1": "str"}}

Here is the EA Forum post to parse:

Title: {title}
Author: {author}

{text}
"""


def extract_claims(
    post: Post, 
    n: int, 
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> List[Claim]:
    """Extract the N most central claims from a post.
    
    Args:
        post: Post object to extract claims from
        n: Number of top claims to extract
        model: Model to use for extraction
        
    Returns:
        List of Claim objects
    """
    text = post.markdown_content or post.html_body or ""
    if not text:
        return []
    if n == 1:
        prompt_content = PROMPT_THESIS_EXTRACTION.format(
            title=post.title,
            author=post.author_display_name or "Unknown",
            text=text
        )
    else: 
        prompt_content = PROMPT_CLAIM_EXTRACTION.format(
            n=n,
            title=post.title,
            author=post.author_display_name or "Unknown",
            text=text
        )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_content}]
    )
    
    raw_content = response.choices[0].message.content
    
    # Strip potential markdown formatting
    if raw_content.startswith("```"):
        raw_content = raw_content.split("```")[1]
        if raw_content.startswith("json"):
            raw_content = raw_content[4:]
        raw_content = raw_content.strip()
    
    claims_json = json.loads(raw_content)
    claims = []
    
    for idx, (key, claim_text) in enumerate(claims_json.items(), 1):
        claim = Claim(
            claim_id=f"{post.post_id}_{idx}",
            post_id=post.post_id,
            claim=claim_text
        )
        claims.append(claim)
    
    return claims