import os
import json
from datetime import datetime
from typing import List, Literal, Optional
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv


from models import Post

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def get_representative_posts(n: Literal[10, 20, 100, 200, 500, 1000]) -> List[Post]:
    assert n in [10, 20, 100, 200, 500, 1000], "n must be 10, 20, 100, 200, 500, or 1000"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT * FROM fellowship_mvp 
                WHERE is_representative_{n} = TRUE
            """)
            return [Post(**row) for row in cur.fetchall()]

def get_posts_by_ids(post_ids: List[str]) -> List[Post]:
    """Get posts by their IDs."""
    if not post_ids:
        return []
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Use parameterized query with IN clause
            placeholders = ','.join(['%s'] * len(post_ids))
            cur.execute(f"""
                SELECT * FROM fellowship_mvp 
                WHERE post_id IN ({placeholders})
            """, post_ids)
            return [Post(**row) for row in cur.fetchall()]

def get_bucketed_sample_posts(n: int) -> List[Post]:
    assert n % 5 == 0, "n must be a multiple of 5"
    posts_per_bucket = n // 5
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                WITH ranked_posts AS (
                    SELECT *,
                        NTILE(5) OVER (ORDER BY base_score) as bucket
                    FROM fellowship_mvp
                    WHERE posted_at >= '2025-01-01' AND posted_at < '2025-08-22'
                )
                SELECT * FROM (
                    SELECT *,
                        ROW_NUMBER() OVER (PARTITION BY bucket ORDER BY MD5(post_id)) as rn
                    FROM ranked_posts
                ) t
                WHERE rn <= {posts_per_bucket}
                ORDER BY bucket, rn
            """)
            return [Post(**row) for row in cur.fetchall()]
        
def save_posts(filename: str, posts: List[Post]):
    with open(filename, 'w') as f:
        json.dump([post.model_dump(mode='json') for post in posts], f, indent=2, default=str)


def read_saved_posts(file_id: str) -> List[Post]:
    file_path = Path(__file__).parent.parent / 'data' / 'raw' / f'{file_id}.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
        return [Post(**post_data) for post_data in data]


def get_chronological_sample_posts(
    n: int, 
    start_datetime: datetime, 
    end_datetime: Optional[datetime] = None
) -> List[Post]:
    """Get every nth post chronologically from a start datetime.
    
    Args:
        n: Take every nth post (e.g., n=10 takes every 10th post)
        start_datetime: Start sampling from this datetime
        end_datetime: Optional end datetime (defaults to latest post)
        
    Returns:
        List of Posts sampled chronologically at regular intervals
        
    Example:
        # Get every 50th post starting from Jan 1, 2024
        posts = get_chronological_sample_posts(
            n=50, 
            start_datetime=datetime(2024, 1, 1)
        )
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Use named parameters to avoid modulo operator issues
            if end_datetime:
                query = """
                    WITH chronological_posts AS (
                        SELECT *,
                               ROW_NUMBER() OVER (ORDER BY posted_at ASC) as row_num
                        FROM fellowship_mvp
                        WHERE posted_at >= %(start_date)s AND posted_at <= %(end_date)s
                        AND posted_at IS NOT NULL
                    )
                    SELECT * FROM chronological_posts
                    WHERE row_num %% %(n_val)s = 1
                    ORDER BY posted_at ASC
                """
                params = {
                    'start_date': start_datetime, 
                    'end_date': end_datetime, 
                    'n_val': n
                }
            else:
                query = """
                    WITH chronological_posts AS (
                        SELECT *,
                               ROW_NUMBER() OVER (ORDER BY posted_at ASC) as row_num
                        FROM fellowship_mvp
                        WHERE posted_at >= %(start_date)s
                        AND posted_at IS NOT NULL
                    )
                    SELECT * FROM chronological_posts
                    WHERE row_num %% %(n_val)s = 1
                    ORDER BY posted_at ASC
                """
                params = {
                    'start_date': start_datetime,
                    'n_val': n
                }
            
            # Get chronologically ordered posts with row numbers
            cur.execute(query, params)
            
            return [Post(**row) for row in cur.fetchall()]


def get_n_most_recent_posts_in_same_cluster(
    post_id: str, 
    cluster_cardinality: Literal[5, 12], 
    n: int
) -> List[Post]:
    """Get the n most recent posts in the same cluster that were posted before the given post.
    
    Args:
        post_id: The ID of the post to find cluster siblings for
        cluster_cardinality: Either 5 or 12 for the cluster grouping
        n: Number of recent posts to return
        
    Returns:
        List of Posts in the same cluster posted before the given post,
        ordered by posted_at descending (most recent first)
    """
    cluster_col = f"ea_cluster_{cluster_cardinality}"
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            # First get the cluster ID and posted_at for the given post
            cur.execute(f"""
                SELECT {cluster_col}, posted_at
                FROM fellowship_mvp
                WHERE post_id = %s
            """, (post_id,))
            
            result = cur.fetchone()
            if not result:
                return []
            
            cluster_id = result[cluster_col]
            post_timestamp = result['posted_at']
            
            # Now get the n most recent posts in the same cluster before this post
            cur.execute(f"""
                SELECT * 
                FROM fellowship_mvp
                WHERE {cluster_col} = %s
                    AND posted_at < %s
                ORDER BY posted_at DESC
                LIMIT %s
            """, (cluster_id, post_timestamp, n))
            
            return [Post(**row) for row in cur.fetchall()]


def get_posts_by_author_in_date_range(
    author_display_name: str,
    start_date: datetime,
    end_date: datetime
) -> List[Post]:
    """Get all posts by an author within a date range.
    
    Args:
        author_display_name: The display name of the author
        start_date: The start date (inclusive)
        end_date: The end date (inclusive)
        
    Returns:
        List of Posts by the author in the date range,
        ordered by posted_at descending (most recent first)
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * 
                FROM fellowship_mvp
                WHERE author_display_name = %s
                    AND posted_at >= %s
                    AND posted_at <= %s
                ORDER BY posted_at DESC
            """, (author_display_name, start_date, end_date))
            
            return [Post(**row) for row in cur.fetchall()]


if __name__ == "__main__":
    
    # representative_10 = get_representative_posts(10)
    # representative_20 = get_representative_posts(20)
    representative_100 = get_representative_posts(100)
    print(f"Found {len(representative_100)} representative posts")
    # 
    # Save to files
    # save_posts('./data/raw/posts/representative_10_2025.json', representative_10)
    os.makedirs('./data/raw/posts', exist_ok=True)
    # save_posts('./data/raw/posts/representative_20_2025.json', representative_20)
    save_posts('./data/raw/posts/representative_100_2025.json', representative_100)
    
