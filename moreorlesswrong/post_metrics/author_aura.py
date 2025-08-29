from typing import Literal
from pydantic import BaseModel
import json
from pathlib import Path
import hashlib
import threading
from collections import defaultdict

from models import Post
from llm_client import client

# Configuration
USE_WEB_SEARCH = False  # True might make more sense but creates data leakage?

# Global locks per author to prevent concurrent computation
author_locks = defaultdict(threading.Lock)


class PostAuthorAura(BaseModel):
    post_id: str
    author_fame_ea: int  # 1-10 fame score within EA community
    author_fame_humanity: int  # 1-10 fame score globally
    explanation: str
    
    @classmethod
    def metric_name(cls) -> str:
        return "PostAuthorAura"
    
    @classmethod
    def metric_score_fields(cls) -> list[str]:
        return ["author_fame_ea", "author_fame_humanity"]
    
    @classmethod
    def human_readable_names(cls) -> dict[str, str]:
        return {
            "author_fame_ea": "Author Fame in EA",
            "author_fame_humanity": "Author Fame Globally"
        }


PROMPT_AUTHOR_FAME = """Evaluate the fame/prominence of this author.

Author (possibly a pseudonym) to evaluate: {author}

Rate their fame on two scales:

1. Within EA/rationalist community (1-10):
   - 1: Unknown in EA circles
   - 3: Occasional contributor
   - 5: Regular contributor, somewhat known
   - 7: Well-known figure, frequent speaker/writer
   - 10: Central figure (e.g., MacAskill, Singer, Ord)

2. Global fame (1-10):
   - 1: No public presence
   - 3: Minor online presence
   - 5: Known in specific professional circles
   - 7: Notable public intellectual/figure
   - 10: Globally famous

Respond with JSON:
{{"author_fame_ea": <int 1-10>, "author_fame_humanity": <int 1-10>, "explanation": "<brief summary of their prominence>"}}
"""
if USE_WEB_SEARCH:
    PROMPT_AUTHOR_FAME += "\n\nYou may use web search to evaluate the fame/prominence of this author."


def get_author_cache_path(author_name: str) -> Path:
    """Get the cache file path for an author's fame scores."""
    # Create a deterministic hash of the author name
    author_hash = hashlib.md5(author_name.encode()).hexdigest()[:8]
    cache_dir = Path("data/author_fame_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{author_hash}_{author_name.replace(' ', '_').replace('/', '_')[:30]}.json"


def load_cached_author_fame(author_name: str) -> dict | None:
    """Load cached fame scores for an author if they exist."""
    cache_path = get_author_cache_path(author_name)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            print(f"  WARNING: Corrupted cache file for {author_name}, deleting")
            cache_path.unlink()  # Delete corrupted file
            return None
    return None


def save_author_fame_cache(author_name: str, fame_data: dict):
    """Save author fame scores to cache."""
    # Validate the data before saving
    if not fame_data or not all(key in fame_data for key in ["author_fame_ea", "author_fame_humanity", "explanation"]):
        print(f"  WARNING: Invalid fame data for {author_name}, not caching")
        return
    
    cache_path = get_author_cache_path(author_name)
    with open(cache_path, 'w') as f:
        json.dump(fame_data, f, indent=2)


def compute_post_author_aura(
    post: Post,
    model: Literal["gpt-5-nano", "gpt-5-mini", "gpt-5"] = "gpt-5-mini"
) -> PostAuthorAura:
    """Compute author fame/aura scores for a post using web search with thread-safe caching.
    
    Args:
        post: The post containing author information
        model: The model to use for evaluation
        
    Returns:
        PostAuthorAura metric object
    """
    author_name = post.author_display_name or "Unknown"
    
    # First check cache without lock (fast path)
    cached_fame = load_cached_author_fame(author_name)
    if cached_fame:
        print(f"  Using cached fame scores for author: {author_name}")
        return PostAuthorAura(
            post_id=post.post_id,
            author_fame_ea=cached_fame["author_fame_ea"],
            author_fame_humanity=cached_fame["author_fame_humanity"],
            explanation=cached_fame["explanation"]
        )
    
    # If not cached, use double-checked locking to prevent concurrent computation
    with author_locks[author_name]:
        # Check cache again inside lock (another thread might have computed it)
        cached_fame = load_cached_author_fame(author_name)
        if cached_fame:
            print(f"  Using cached fame scores for author (computed by another thread): {author_name}")
            return PostAuthorAura(
                post_id=post.post_id,
                author_fame_ea=cached_fame["author_fame_ea"],
                author_fame_humanity=cached_fame["author_fame_humanity"],
                explanation=cached_fame["explanation"]
            )
        
        # Still not cached, compute fame scores
        print(f"  Computing fame scores for author: {author_name}")
        
        prompt = PROMPT_AUTHOR_FAME.format(author=author_name)
        
        # Use GPT-5 with or without web search capability
        if USE_WEB_SEARCH:
            response = client.responses.create(
                model=model,
                tools=[{"type": "web_search_preview"}],
                input=prompt
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
        
        # Get response content based on API used
        if USE_WEB_SEARCH:
            raw_content = response.output_text
        else:
            raw_content = response.choices[0].message.content
        
        # Strip potential markdown formatting
        if raw_content.startswith("```"):
            raw_content = raw_content.split("```")[1]
            if raw_content.startswith("json"):
                raw_content = raw_content[4:]
            raw_content = raw_content.strip()
        
        result = json.loads(raw_content)
        
        # Cache the result for this author
        fame_data = {
            "author_fame_ea": result["author_fame_ea"],
            "author_fame_humanity": result["author_fame_humanity"],
            "explanation": result["explanation"]
        }
        save_author_fame_cache(author_name, fame_data)
        
        return PostAuthorAura(
            post_id=post.post_id,
            author_fame_ea=result["author_fame_ea"],
            author_fame_humanity=result["author_fame_humanity"],
            explanation=result["explanation"]
        )