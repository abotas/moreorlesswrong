"""Utilities for parsing JSON with error handling and repair."""

import json
from json_repair import repair_json


def parse_json_with_repair(raw_content: str) -> dict:
    """Parse JSON string with automatic repair for common issues.
    
    Args:
        raw_content: Raw JSON string that may need repair
        
    Returns:
        Parsed JSON as dictionary
        
    Raises:
        ValueError: If JSON cannot be parsed even after repair
    """
    # Strip potential markdown formatting
    if raw_content.startswith("```"):
        raw_content = raw_content.split("```")[1]
        if raw_content.startswith("json"):
            raw_content = raw_content[4:]
        raw_content = raw_content.strip()
    
    # First try normal JSON parsing
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        # If that fails, try repairing the JSON
        try:
            repaired = repair_json(raw_content)
            return json.loads(repaired)
        except Exception as e:
            raise ValueError(f"Could not parse JSON even after repair. Original: {raw_content[:200]}...") from e