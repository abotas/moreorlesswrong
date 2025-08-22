from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


class Post(BaseModel):
    id: int
    post_id: str
    title: str
    title_normalized: Optional[str] = None
    page_url: Optional[str] = None
    html_body: Optional[str] = None
    base_score: Optional[int] = None
    comment_count: Optional[int] = 0
    posted_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    author_id: Optional[str] = None
    author_display_name: Optional[str] = None
    coauthor_ids: Optional[List[str]] = None
    coauthor_names: Optional[List[str]] = None
    tag_ids: Optional[List[str]] = None
    tag_names: Optional[List[str]] = None
    markdown_content: Optional[str] = None
    word_count: Optional[int] = None
    reading_time_minutes: Optional[int] = None
    external_links: Optional[List[str]] = None
    short_summary: Optional[str] = None
    long_summary: Optional[str] = None
    source_type: Optional[str] = "EA Forum"
    processing_version: Optional[str] = "1.0"
    processing_errors: Optional[str] = None
    scraped_at: Optional[datetime] = None


class Comment(BaseModel):
    id: int
    comment_id: str
    post_id: str
    parent_comment_id: Optional[str] = None
    html_body: Optional[str] = None
    markdown_content: Optional[str] = None
    base_score: Optional[int] = None
    word_count: Optional[int] = None
    author_id: Optional[str] = None
    author_display_name: Optional[str] = None
    author_slug: Optional[str] = None
    posted_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    source_type: Optional[str] = "EA Forum"
    processing_version: Optional[str] = "1.0"
    processing_errors: Optional[str] = None


class Claim(BaseModel):
    claim_id: str
    post_id: str
    claim: str