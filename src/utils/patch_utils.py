"""
Utilities for processing patch files
"""

import re
from pathlib import Path
from typing import Dict, Optional, Set, List, Tuple


def parse_patch_file(patch_path: str) -> Dict[str, int]:
    """
    Parse a patch file and extract statistics
    
    Args:
        patch_path: Path to the patch file
        
    Returns:
        Dictionary containing:
        - added_lines: Number of lines added
        - deleted_lines: Number of lines deleted
        - patch_size: Total lines changed (added + deleted)
    """
    patch_file = Path(patch_path)
    
    if not patch_file.exists():
        return {
            'added_lines': 0,
            'deleted_lines': 0,
            'patch_size': 0
        }
    
    added_lines = 0
    deleted_lines = 0
    
    with open(patch_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Count lines that start with + (but not +++)
            if line.startswith('+') and not line.startswith('+++'):
                added_lines += 1
            # Count lines that start with - (but not ---)
            elif line.startswith('-') and not line.startswith('---'):
                deleted_lines += 1
    
    return {
        'added_lines': added_lines,
        'deleted_lines': deleted_lines,
        'patch_size': added_lines + deleted_lines
    }


def get_patch_size(patch_path: str) -> int:
    """
    Get patch size (added + deleted lines)
    
    Args:
        patch_path: Path to the patch file
        
    Returns:
        Total number of lines changed
    """
    stats = parse_patch_file(patch_path)
    return stats['patch_size']


def get_modified_files_from_patch(patch_path: str) -> int:
    """
    Count number of modified files in a patch
    
    Args:
        patch_path: Path to the patch file
        
    Returns:
        Number of modified files
    """
    patch_file = Path(patch_path)
    
    if not patch_file.exists():
        return 0
    
    modified_files = set()
    
    with open(patch_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Match diff --git a/file b/file
            if line.startswith('diff --git'):
                match = re.search(r'a/(.*?)\s+b/', line)
                if match:
                    modified_files.add(match.group(1))
    
    return len(modified_files)


def extract_modified_files_and_lines(patch_content: str) -> Dict[str, Set[int]]:
    """
    Extract modified files and line numbers from patch content
    
    Args:
        patch_content: Patch content as string
        
    Returns:
        Dictionary mapping file path to set of modified line numbers
    """
    file_lines = {}
    current_file = None
    current_line_num = 0
    
    lines = patch_content.split('\n')
    
    for line in lines:
        # Match diff --git a/file b/file
        if line.startswith('diff --git'):
            match = re.search(r'diff --git a/(.*?) b/\1', line)
            if not match:
                # Fallback: just extract from b/ part
                match = re.search(r' b/(.+)$', line)
            if match:
                current_file = match.group(1).strip()
                file_lines[current_file] = set()
        
        # Match hunk header: @@ -old_start,old_count +new_start,new_count @@
        elif line.startswith('@@'):
            match = re.search(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
            if match:
                current_line_num = int(match.group(1))
        
        # Process modified lines
        elif current_file is not None and current_line_num > 0:
            if line.startswith('+') and not line.startswith('+++'):
                # Added line
                file_lines[current_file].add(current_line_num)
                current_line_num += 1
            elif line.startswith('-') and not line.startswith('---'):
                # Deleted line (don't increment line number)
                file_lines[current_file].add(current_line_num)
            elif not line.startswith('\\'):
                # Context line (no + or -)
                current_line_num += 1
    
    return file_lines


def parse_patch_content(patch_content: str) -> Dict[str, int]:
    """
    Parse patch content string and extract statistics
    
    Args:
        patch_content: Patch content as string
        
    Returns:
        Dictionary containing added_lines, deleted_lines, patch_size
    """
    added_lines = 0
    deleted_lines = 0
    
    for line in patch_content.split('\n'):
        if line.startswith('+') and not line.startswith('+++'):
            added_lines += 1
        elif line.startswith('-') and not line.startswith('---'):
            deleted_lines += 1
    
    return {
        'added_lines': added_lines,
        'deleted_lines': deleted_lines,
        'patch_size': added_lines + deleted_lines
    }


def get_modified_files_set(patch_content: str) -> Set[str]:
    """
    Get set of modified files from patch content
    
    Args:
        patch_content: Patch content as string
        
    Returns:
        Set of modified file paths
    """
    modified_files = set()
    
    for line in patch_content.split('\n'):
        if line.startswith('diff --git'):
            # Match: diff --git a/file b/file
            match = re.search(r'diff --git a/(.*?) b/\1', line)
            if not match:
                # Fallback: just extract from b/ part
                match = re.search(r' b/(.+)$', line)
            if match:
                modified_files.add(match.group(1).strip())
    
    return modified_files


def calculate_file_coverage(agent_patch: str, golden_patch: str) -> float:
    """
    Calculate file-level coverage between agent and golden patches
    
    File Coverage = |F_gen ∩ F_ref| / |F_ref|
    
    Args:
        agent_patch: Agent-generated patch content
        golden_patch: Golden reference patch content
        
    Returns:
        File coverage ratio (0.0 to 1.0)
    """
    f_gen = get_modified_files_set(agent_patch)
    f_ref = get_modified_files_set(golden_patch)
    
    if len(f_ref) == 0:
        return 0.0
    
    intersection = f_gen & f_ref
    return len(intersection) / len(f_ref)


def calculate_line_coverage(agent_patch: str, golden_patch: str) -> float:
    """
    Calculate line-level coverage between agent and golden patches
    
    Line Coverage = Σ|L_gen^f ∩ L_ref^f| / Σ|L_ref^f|
    
    Args:
        agent_patch: Agent-generated patch content
        golden_patch: Golden reference patch content
        
    Returns:
        Line coverage ratio (0.0 to 1.0)
    """
    l_gen = extract_modified_files_and_lines(agent_patch)
    l_ref = extract_modified_files_and_lines(golden_patch)
    
    total_ref_lines = 0
    total_intersect_lines = 0
    
    # Only consider files in reference patch
    for file in l_ref:
        ref_lines = l_ref[file]
        gen_lines = l_gen.get(file, set())
        
        total_ref_lines += len(ref_lines)
        total_intersect_lines += len(ref_lines & gen_lines)
    
    if total_ref_lines == 0:
        return 0.0
    
    return total_intersect_lines / total_ref_lines
