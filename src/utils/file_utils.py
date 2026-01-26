"""
Provides common file operations and filtering functions.
"""

from pathlib import Path
from typing import Union


def is_java_test_file(file_path: Union[str, Path]) -> bool:
    """
    Determines whether a file is a Java test file.
    
    Criteria:
    1. File path contains 'src/test/' (Maven standard test directory structure)
    2. File path contains 'src/androidTest/' (Android test directory)
    3. File path contains '/test/' or '/tests/' directory
    4. File name ends with 'Test' (e.g., UserTest.java)
    5. File name starts with 'test' (e.g., testUser.java)
    6. File name contains 'Test' and ends with '.java' (e.g., UserTestCase.java, TestUtils.java)
    
    Args:
        file_path: File path (string or Path object)
    
    Returns:
        True if it is a Java test file, False otherwise
    
    Examples:
        >>> is_java_test_file("src/test/java/com/example/UserTest.java")
        True
        >>> is_java_test_file("src/main/java/com/example/User.java")
        False
        >>> is_java_test_file("com/example/tests/IntegrationTest.java")
        True
        >>> is_java_test_file("UserTest.java")
        True
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    # Convert to POSIX path string for unified processing
    path_str = file_path.as_posix()
    file_name = file_path.name
    
    # 1. Check if in standard test directories
    test_directories = [
        'src/test/',
        'src/androidTest/',
        '/test/',
        '/tests/',
    ]
    
    for test_dir in test_directories:
        if test_dir in path_str:
            return True
    
    # 2. Check file name patterns (only for .java files)
    if not file_name.endswith('.java'):
        return False
    
    # Remove .java suffix
    name_without_ext = file_name[:-5]
    
    # File name ends with Test
    if name_without_ext.endswith('Test'):
        return True
    
    # File name starts with test
    if name_without_ext.startswith('test'):
        return True
    
    # File name contains Test (e.g., TestUtils, UserTestCase)
    if 'Test' in name_without_ext:
        return True
    
    return False


def filter_non_test_refactorings(refactoring_data: dict) -> dict:
    """
    Filters out refactorings in test files from RefactoringMiner output.
    
    Args:
        refactoring_data: JSON data output by RefactoringMiner (dict format)
    
    Returns:
        Filtered refactoring data with the same format as input
    
    Examples:
        Input refactoring_data format:
        {
            "commits": [
                {
                    "repository": "...",
                    "sha1": "...",
                    "refactorings": [
                        {
                            "type": "Rename Method",
                            "leftSideLocations": [...],
                            "rightSideLocations": [...]
                        }
                    ]
                }
            ]
        }
    """
    if not refactoring_data or 'commits' not in refactoring_data:
        return refactoring_data
    
    filtered_data = {
        'commits': []
    }
    
    for commit in refactoring_data.get('commits', []):
        filtered_refactorings = []
        
        for refactoring in commit.get('refactorings', []):
            # Check all file locations on left and right sides
            is_test = False
            
            # Check left side locations
            for location in refactoring.get('leftSideLocations', []):
                file_path = location.get('filePath', '')
                if is_java_test_file(file_path):
                    is_test = True
                    break
            
            # Check right side locations
            if not is_test:
                for location in refactoring.get('rightSideLocations', []):
                    file_path = location.get('filePath', '')
                    if is_java_test_file(file_path):
                        is_test = True
                        break
            
            # Keep it if not a refactoring in test files
            if not is_test:
                filtered_refactorings.append(refactoring)
        
        # Create new commit object (only add when there are non-test refactorings)
        if filtered_refactorings:
            filtered_commit = commit.copy()
            filtered_commit['refactorings'] = filtered_refactorings
            filtered_data['commits'].append(filtered_commit)
    
    return filtered_data