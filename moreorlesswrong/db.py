import os
import json
from datetime import datetime
from typing import List, Literal
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv


from models import Post

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def get_representative_posts(n: Literal[10, 20, 100]) -> List[Post]:
    assert n in [10, 20, 100], "n must be 10, 20, or 100"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT * FROM fellowship_mvp 
                WHERE is_representative_{n} = TRUE
            """)
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


if __name__ == "__main__":
    
    # representative_10 = get_representative_posts(10)
    representative_20 = get_representative_posts(20)
    print(f"Found {len(representative_20)} representative posts")
    # representative_100 = get_representative_posts(100)
    # 
    # Save to files
    # save_posts('./data/raw/posts/representative_10_2025.json', representative_10)
    os.makedirs('./data/raw/posts', exist_ok=True)
    save_posts('./data/raw/posts/representative_20_2025.json', representative_20)
    # save_posts('./data/raw/posts/representative_100_2025.json', representative_100)
    
