"""
This file contains the constants for the project.
"""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REPOS_DIR = ROOT_DIR / "repos"

MSB_DIR = DATA_DIR / "Multi-SWE-bench"
MSB_JAVA_DIR = MSB_DIR / "java"

JAVA_EXPERIMENT_DIR = DATA_DIR / "java_experiment"

# Output directory for patch application results
PATCH_APPLY_OUTPUT_DIR = DATA_DIR / "patch_apply_results"

# RefactoringMiner related
REFACTORING_MINER_PATH = ROOT_DIR / "tools" / "RefactoringMiner-3.0.13" / "bin" / "RefactoringMiner"
REFACTORING_DETECTION_TIMEOUT = 300  # 5 minutes

# Output directory for refactoring detection results
REFACTORING_DETECTION_OUTPUT_DIR = DATA_DIR / "refactoring_detection_results"
AGENT_REFACTORING_OUTPUT_DIR = REFACTORING_DETECTION_OUTPUT_DIR / "agent"
GOLDEN_REFACTORING_OUTPUT_DIR = REFACTORING_DETECTION_OUTPUT_DIR / "golden"

# Evaluation logs directory (for compilation and final reports)
EVALUATION_LOGS_DIR = DATA_DIR / "evaluation_logs_new"

TAG_PREFIX = "agent_patch"
