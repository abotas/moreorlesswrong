"""Helper functions for formatting raw related posts context when bypassing synthesizer."""

from typing import List
from models import Post


def format_raw_related_posts(related_posts: List[Post]) -> str:
    """Format related posts with raw 20k character previews.
    
    This is used when bypassing the synthesizer agent to directly include
    the first 20k characters of each related post instead of using
    intelligent context synthesis.
    
    Args:
        related_posts: List of related posts to format
        
    Returns:
        Formatted string with raw content previews
    """
    if not related_posts:
        return "No related posts available for context."
    
    formatted_posts = []
    
    for i, post in enumerate(related_posts, 1):
        # Get post content (markdown preferred, fallback to HTML)
        post_content = post.markdown_content or post.html_body or "No content available"
        
        # Take first 20k characters
        content_preview = post_content[:20_000]
        if len(post_content) > 20_000:
            content_preview += "..."
        
        # Format with title, author, karma, and content
        author_name = post.author_display_name or "Unknown"
        karma = post.base_score or 0
        
        formatted_post = f"""
Related Post #{i}: "{post.title}" by {author_name} (Karma: {karma})
Content:
```
{content_preview}
```
"""
        formatted_posts.append(formatted_post)
    
    return "\n".join(formatted_posts)