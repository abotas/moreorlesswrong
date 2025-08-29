#!/usr/bin/env python3
"""
Safe import script for alej_v1 metrics from JSON files to database.
Includes validation, dry-run mode, and rollback capabilities.
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from dotenv import load_dotenv
import argparse
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

METRICS_DIR = Path(__file__).parent.parent / "data" / "post_metrics" / "mini1000"

# Define which metrics to extract (excluding explanation fields)
METRIC_MAPPING = {
    "PostValue": ["value_ea", "value_humanity"],
    "PostRobustness": ["robustness_score"],
    "PostAuthorAura": ["author_fame_ea", "author_fame_humanity"],
    "PostClarity": ["clarity_score"],
    "PostNovelty": ["novelty_ea", "novelty_humanity"],
    "PostInferentialSupport": ["reasoning_quality", "evidence_quality", "overall_support"],
    "PostExternalValidation": ["emperical_claim_validation_score"]  # Note: typo in original
}


def get_connection():
    """Get database connection with RealDictCursor."""
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def extract_metrics_from_json(json_path: Path) -> Optional[Dict]:
    """Extract flattened metrics from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract post_id from first available metric group
        post_id = None
        for key in data:
            if 'post_id' in data[key]:
                post_id = data[key]['post_id']
                break
        
        if not post_id:
            print(f"Warning: No post_id found in {json_path}")
            return None
        
        # Extract and flatten metrics
        flattened_metrics = {}
        for group_name, field_names in METRIC_MAPPING.items():
            if group_name in data:
                for field in field_names:
                    if field in data[group_name]:
                        flattened_metrics[field] = data[group_name][field]
        
        return {
            "post_id": post_id,
            "metrics": flattened_metrics
        }
    
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None


def validate_metrics(metrics: Dict) -> bool:
    """Validate that metrics have expected fields and types."""
    expected_fields = [
        "value_ea", "value_humanity", "robustness_score",
        "author_fame_ea", "author_fame_humanity", "clarity_score",
        "novelty_ea", "novelty_humanity", "reasoning_quality",
        "evidence_quality", "overall_support", "emperical_claim_validation_score"
    ]
    
    # Check if we have at least some expected fields
    present_fields = [f for f in expected_fields if f in metrics]
    if len(present_fields) < 6:  # Require at least half the fields
        return False
    
    # Validate that numeric fields are actually numbers
    for field, value in metrics.items():
        if field in expected_fields and not isinstance(value, (int, float)):
            return False
    
    return True


def check_post_exists(conn, post_id: str) -> bool:
    """Check if a post exists in the database."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM fellowship_mvp WHERE post_id = %s", (post_id,))
        return cur.fetchone() is not None


def import_metrics(dry_run: bool = True, limit: Optional[int] = None, single_post: Optional[str] = None):
    """
    Import metrics from JSON files to database.
    
    Args:
        dry_run: If True, only simulate the import without making changes
        limit: Limit number of files to process (for testing)
        single_post: Process only a specific post_id (for testing)
    """
    # Collect all JSON files
    json_files = list(METRICS_DIR.glob("*.json"))
    
    if single_post:
        json_files = [f for f in json_files if f.stem == single_post]
        if not json_files:
            print(f"Error: No JSON file found for post_id {single_post}")
            return
    
    if limit:
        json_files = json_files[:limit]
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process each file
    successful_imports = 0
    failed_imports = 0
    posts_not_found = 0
    
    conn = get_connection()
    
    try:
        for json_path in tqdm(json_files, desc="Processing files"):
            # Extract metrics
            result = extract_metrics_from_json(json_path)
            if not result:
                failed_imports += 1
                continue
            
            post_id = result["post_id"]
            metrics = result["metrics"]
            
            # Validate metrics
            if not validate_metrics(metrics):
                print(f"Warning: Invalid metrics for {post_id}, skipping")
                failed_imports += 1
                continue
            
            # Check if post exists
            if not check_post_exists(conn, post_id):
                posts_not_found += 1
                if not dry_run:
                    print(f"Warning: Post {post_id} not found in database, skipping")
                continue
            
            # Update database (or simulate)
            if not dry_run:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE fellowship_mvp 
                        SET alej_v1_metrics = %s,
                            alej_v1_metrics_imported_at = %s
                        WHERE post_id = %s
                    """, (Json(metrics), datetime.now(), post_id))
                    conn.commit()
            
            successful_imports += 1
        
    except Exception as e:
        print(f"Error during import: {e}")
        if not dry_run:
            conn.rollback()
            print("Transaction rolled back")
        raise
    finally:
        conn.close()
    
    # Print summary
    print("\n" + "="*50)
    print("Import Summary:")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"  Successfully processed: {successful_imports}")
    print(f"  Failed to process: {failed_imports}")
    print(f"  Posts not found in DB: {posts_not_found}")
    print(f"  Total files: {len(json_files)}")
    
    if dry_run:
        print("\n⚠️  This was a dry run. No changes were made to the database.")
        print("Run with --execute to perform the actual import.")


def verify_import():
    """Verify that metrics were imported correctly."""
    conn = get_connection()
    
    with conn.cursor() as cur:
        # Count posts with metrics
        cur.execute("""
            SELECT COUNT(*) as total,
                   COUNT(alej_v1_metrics) as with_metrics
            FROM fellowship_mvp
            WHERE post_id IN (
                SELECT DISTINCT post_id 
                FROM fellowship_mvp 
                WHERE is_representative_1000 = TRUE
            )
        """)
        result = cur.fetchone()
        
        print("\nVerification Results:")
        print(f"  Posts in representative_1000: {result['total']}")
        print(f"  Posts with metrics: {result['with_metrics']}")
        
        # Sample a few metrics
        cur.execute("""
            SELECT post_id, 
                   alej_v1_metrics->>'value_ea' as value_ea,
                   alej_v1_metrics->>'clarity_score' as clarity_score
            FROM fellowship_mvp
            WHERE alej_v1_metrics IS NOT NULL
            LIMIT 5
        """)
        
        print("\nSample imported metrics:")
        for row in cur.fetchall():
            print(f"  {row['post_id']}: value_ea={row['value_ea']}, clarity={row['clarity_score']}")
    
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Import alej_v1 metrics to database")
    parser.add_argument("--execute", action="store_true", 
                       help="Actually perform the import (default is dry-run)")
    parser.add_argument("--limit", type=int, 
                       help="Limit number of files to process")
    parser.add_argument("--single-post", type=str,
                       help="Process only a specific post_id")
    parser.add_argument("--verify", action="store_true",
                       help="Verify import results")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_import()
    else:
        import_metrics(
            dry_run=not args.execute,
            limit=args.limit,
            single_post=args.single_post
        )


if __name__ == "__main__":
    main()