"""
Utilities for processing refactoring detection results
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter


def load_refactoring_json(json_path: str) -> Dict:
    """
    Load refactoring detection result from JSON file
    
    Args:
        json_path: Path to refactoring JSON file
        
    Returns:
        Dictionary containing refactoring data
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_refactorings(refactoring_data: Dict) -> List[Dict]:
    """
    Extract refactorings from loaded JSON data
    
    Args:
        refactoring_data: Loaded refactoring JSON data
        
    Returns:
        List of refactoring dictionaries
    """
    refactorings = []
    
    if 'commits' in refactoring_data:
        for commit in refactoring_data['commits']:
            if 'refactorings' in commit:
                refactorings.extend(commit['refactorings'])
    
    return refactorings


def count_refactoring_types(refactorings: List[Dict]) -> Dict[str, int]:
    """
    Count occurrences of each refactoring type
    
    Args:
        refactorings: List of refactoring dictionaries
        
    Returns:
        Dictionary mapping refactoring type to count
    """
    types = [r['type'] for r in refactorings if 'type' in r]
    return dict(Counter(types))


def has_refactoring(refactorings: List[Dict]) -> bool:
    """
    Check if any refactorings exist
    
    Args:
        refactorings: List of refactoring dictionaries
        
    Returns:
        True if refactorings exist, False otherwise
    """
    return len(refactorings) > 0


def convert_to_refactoring_info(refactoring: Dict) -> Dict:
    """
    Convert raw refactoring dict to RefactoringInfo format
    
    Args:
        refactoring: Raw refactoring dictionary
        
    Returns:
        Dictionary in RefactoringInfo format
    """
    return {
        'type': refactoring.get('type', ''),
        'description': refactoring.get('description', ''),
        'left_locations': refactoring.get('leftSideLocations', []),
        'right_locations': refactoring.get('rightSideLocations', [])
    }


def process_refactoring_file(json_path: str) -> Tuple[bool, Dict[str, int], List[Dict]]:
    """
    Process a refactoring JSON file and extract all information
    
    Args:
        json_path: Path to refactoring JSON file
        
    Returns:
        Tuple of (has_refactoring, type_counts, refactoring_list)
    """
    if not Path(json_path).exists():
        return False, {}, []
    
    data = load_refactoring_json(json_path)
    refactorings = extract_refactorings(data)
    
    has_ref = has_refactoring(refactorings)
    type_counts = count_refactoring_types(refactorings)
    refactoring_list = [convert_to_refactoring_info(r) for r in refactorings]
    
    return has_ref, type_counts, refactoring_list
