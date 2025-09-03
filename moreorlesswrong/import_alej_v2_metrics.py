#!/usr/bin/env python3
"""
Safe import script for alej_v2 metrics from JSON files to database.
Includes validation, dry-run mode, and rollback capabilities.
Based on the v2 metrics currently enabled in post_metric_pipeline.py.
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

# Default to v2-mini directory, but allow override
DEFAULT_METRICS_DIR = Path(__file__).parent.parent / "data" / "post_metrics" / "v2-mini"

# Define which v2 metrics to extract (based on currently enabled metrics in pipeline)
V2_METRIC_MAPPING = {
    "ValueV2": ["value_score"],
    "CooperativenessV2": ["cooperativeness_score"],
    "ClarityV2": ["clarity_score"],
    "PrecisionV2": ["precision_score"],
    "AuthorAuraV2": ["ea_fame_score"],
    "ExternalValidationV2": ["external_validation_score"],
    "RobustnessV2": ["robustness_score"],
    "ReasoningQualityV2": ["reasoning_quality_score"],
    "TitleClickabilityV2": ["title_clickability_score"],
    "ControversyTemperatureV2": ["controversy_temperature_score"],
    "EmpiricalEvidenceQualityV2": ["empirical_evidence_quality_score"]
}


def get_connection():
    """Get database connection with RealDictCursor."""
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def extract_metrics_from_json(json_path: Path) -> Optional[Dict]:
    """Extract flattened v2 metrics from a JSON file."""
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
        
        # Extract and flatten v2 metrics
        flattened_metrics = {}
        for group_name, field_names in V2_METRIC_MAPPING.items():
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


def validate_v2_metrics(metrics: Dict) -> bool:
    """Validate that v2 metrics have expected fields and types."""
    expected_fields = [
        "value_score", "cooperativeness_score", "clarity_score", "precision_score",
        "ea_fame_score", "external_validation_score", "robustness_score",
        "reasoning_quality_score", "title_clickability_score", 
        "controversy_temperature_score", "empirical_evidence_quality_score"
    ]
    
    # Check if we have at least some expected fields
    present_fields = [f for f in expected_fields if f in metrics]
    if len(present_fields) < 5:  # Require at least 5 fields present
        return False
    
    # Validate that numeric fields are actually numbers (1-10 scale)
    for field, value in metrics.items():
        if field in expected_fields:
            if not isinstance(value, (int, float)) or not (1 <= value <= 10):
                print(f"Warning: Invalid value for {field}: {value} (expected 1-10)")
                return False
    
    return True


def ensure_columns_exist(conn):
    """Create alej_v2_metrics columns if they don't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE fellowship_mvp 
            ADD COLUMN IF NOT EXISTS alej_v2_metrics JSONB,
            ADD COLUMN IF NOT EXISTS alej_v2_metrics_imported_at TIMESTAMP
        """)
        conn.commit()
        print("✓ Ensured alej_v2_metrics columns exist")


def check_post_exists(conn, post_id: str) -> bool:
    """Check if a post exists in the database."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM fellowship_mvp WHERE post_id = %s", (post_id,))
        return cur.fetchone() is not None


def import_v2_metrics(dry_run: bool = True, limit: Optional[int] = None, 
                     single_post: Optional[str] = None, metrics_dir: Optional[str] = None):
    """
    Import v2 metrics from JSON files to database.
    
    Args:
        dry_run: If True, only simulate the import without making changes
        limit: Limit number of files to process (for testing)
        single_post: Process only a specific post_id (for testing)
        metrics_dir: Custom directory path for metrics (overrides default)
    """
    # Use custom directory if provided, otherwise use default
    if metrics_dir:
        metrics_path = Path(metrics_dir)
    else:
        metrics_path = DEFAULT_METRICS_DIR
    
    if not metrics_path.exists():
        print(f"Error: Metrics directory {metrics_path} does not exist")
        return
    
    # Collect all JSON files
    json_files = list(metrics_path.glob("*.json"))
    
    if single_post:
        json_files = [f for f in json_files if f.stem == single_post]
        if not json_files:
            print(f"Error: No JSON file found for post_id {single_post}")
            return
    
    if limit:
        json_files = json_files[:limit]
    
    print(f"Found {len(json_files)} JSON files to process in {metrics_path}")
    
    # Process each file
    successful_imports = 0
    failed_imports = 0
    posts_not_found = 0
    
    conn = get_connection()
    
    try:
        # Ensure the necessary columns exist
        ensure_columns_exist(conn)
        
        for json_path in tqdm(json_files, desc="Processing v2 metrics files"):
            # Extract metrics
            result = extract_metrics_from_json(json_path)
            if not result:
                failed_imports += 1
                continue
            
            post_id = result["post_id"]
            metrics = result["metrics"]
            
            # Validate metrics
            if not validate_v2_metrics(metrics):
                print(f"Warning: Invalid v2 metrics for {post_id}, skipping")
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
                        SET alej_v2_metrics = %s,
                            alej_v2_metrics_imported_at = %s
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
    print("V2 Import Summary:")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"  Metrics directory: {metrics_path}")
    print(f"  Successfully processed: {successful_imports}")
    print(f"  Failed to process: {failed_imports}")
    print(f"  Posts not found in DB: {posts_not_found}")
    print(f"  Total files: {len(json_files)}")
    
    print(f"\n  Enabled v2 metrics ({len(V2_METRIC_MAPPING)}):")
    for metric_name, fields in V2_METRIC_MAPPING.items():
        print(f"    {metric_name}: {', '.join(fields)}")
    
    if dry_run:
        print("\n⚠️  This was a dry run. No changes were made to the database.")
        print("Run with --execute to perform the actual import.")


def verify_v2_import():
    """Verify that v2 metrics were imported correctly."""
    conn = get_connection()
    
    with conn.cursor() as cur:
        # Count posts with v2 metrics
        cur.execute("""
            SELECT COUNT(*) as total,
                   COUNT(alej_v2_metrics) as with_v2_metrics
            FROM fellowship_mvp
        """)
        result = cur.fetchone()
        
        print("\nV2 Verification Results:")
        print(f"  Total posts in DB: {result['total']}")
        print(f"  Posts with v2 metrics: {result['with_v2_metrics']}")
        
        # Sample a few v2 metrics
        cur.execute("""
            SELECT post_id, 
                   alej_v2_metrics->>'value_score' as value_score,
                   alej_v2_metrics->>'clarity_score' as clarity_score,
                   alej_v2_metrics->>'title_clickability_score' as title_clickability,
                   alej_v2_metrics_imported_at
            FROM fellowship_mvp
            WHERE alej_v2_metrics IS NOT NULL
            ORDER BY alej_v2_metrics_imported_at DESC
            LIMIT 5
        """)
        
        print("\nSample imported v2 metrics (most recent):")
        for row in cur.fetchall():
            print(f"  {row['post_id']}: value={row['value_score']}, "
                  f"clarity={row['clarity_score']}, clickability={row['title_clickability']}")
        
        # Check field coverage
        print("\nField coverage analysis:")
        for metric_name, field_names in V2_METRIC_MAPPING.items():
            for field in field_names:
                cur.execute(f"""
                    SELECT COUNT(*) as count
                    FROM fellowship_mvp
                    WHERE alej_v2_metrics->>'{field}' IS NOT NULL
                """)
                count = cur.fetchone()['count']
                print(f"  {field}: {count} posts")
    
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Import alej_v2 metrics to database")
    parser.add_argument("--execute", action="store_true", 
                       help="Actually perform the import (default is dry-run)")
    parser.add_argument("--limit", type=int, 
                       help="Limit number of files to process")
    parser.add_argument("--single-post", type=str,
                       help="Process only a specific post_id")
    parser.add_argument("--metrics-dir", type=str,
                       help="Custom metrics directory (default: data/post_metrics/v2-mini)")
    parser.add_argument("--verify", action="store_true",
                       help="Verify import results")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_v2_import()
    else:
        import_v2_metrics(
            dry_run=not args.execute,
            limit=args.limit,
            single_post=args.single_post,
            metrics_dir=args.metrics_dir
        )


if __name__ == "__main__":
    main()