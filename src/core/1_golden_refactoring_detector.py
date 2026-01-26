"""
Detect refactorings from golden patches (official PRs) using RefactoringMiner CLI.

Note: fix_patch only contains source code changes, not test file changes.
Therefore, no filtering is needed.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

# Add project root to path for direct script execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.constant import (
    GOLDEN_REFACTORING_OUTPUT_DIR,
    MSB_JAVA_DIR,
    REFACTORING_DETECTION_TIMEOUT,
    REFACTORING_MINER_PATH,
    REPOS_DIR,
)


@dataclass
class GoldenInstance:
    """Information about a golden instance (PR) to be analyzed."""
    instance_id: str
    repo: str
    base_sha: str
    fix_patch: str
    pr_number: int


def setup_logging(log_path: Path) -> None:
    """Setup logging configuration."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def run_cmd(
    args: List[str],
    cwd: Path,
    check: bool = True,
    capture_output: bool = True,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    return subprocess.run(
        args,
        cwd=str(cwd),
        check=check,
        text=True,
        capture_output=capture_output,
        timeout=timeout,
    )


class GoldenRefactoringDetector:
    """Detect refactorings from golden patches (official PRs)."""
    
    def __init__(
        self,
        refactoring_miner_path: Path = REFACTORING_MINER_PATH,
        repos_dir: Path = REPOS_DIR,
        golden_data_dir: Path = MSB_JAVA_DIR,
        output_dir: Path = GOLDEN_REFACTORING_OUTPUT_DIR,
        timeout: int = REFACTORING_DETECTION_TIMEOUT,
    ):
        self.refactoring_miner_path = refactoring_miner_path
        self.repos_dir = repos_dir
        self.golden_data_dir = golden_data_dir
        self.output_dir = output_dir
        self.timeout = timeout
        
        if not self.refactoring_miner_path.exists():
            raise FileNotFoundError(f"RefactoringMiner not found at {self.refactoring_miner_path}")
    
    def load_golden_instances(self) -> List[GoldenInstance]:
        """Load golden instances from Multi-SWE-bench Java dataset."""
        instances = []
        
        for jsonl_file in sorted(self.golden_data_dir.glob("*.jsonl")):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    data = json.loads(line)
                    
                    instance_id = data.get("instance_id")
                    repo = data.get("repo")
                    base = data.get("base", {})
                    base_sha = base.get("sha")
                    fix_patch = data.get("fix_patch", "")
                    pr_number = data.get("number")
                    
                    # Skip instances without required data
                    if not all([instance_id, repo, base_sha, fix_patch]):
                        logging.warning(f"Skipping instance with missing data: {instance_id}")
                        continue
                    
                    instances.append(
                        GoldenInstance(
                            instance_id=instance_id,
                            repo=repo,
                            base_sha=base_sha,
                            fix_patch=fix_patch,
                            pr_number=pr_number or 0,
                        )
                    )
        
        logging.info(f"Loaded {len(instances)} golden instances from {self.golden_data_dir}")
        return instances
    
    def detect_single_instance(self, instance: GoldenInstance, output_json: Path) -> Dict:
        """Detect refactorings for a single golden instance."""
        repo_path = self.repos_dir / instance.repo
        
        if not repo_path.exists():
            return {
                "instance_id": instance.instance_id,
                "status": "failed",
                "error": f"Repository not found: {repo_path}",
            }
        
        # Get current branch/commit to restore later
        try:
            current_ref_result = run_cmd(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_path,
                check=False,
            )
            if current_ref_result.returncode == 0 and current_ref_result.stdout.strip() != "HEAD":
                original_ref = current_ref_result.stdout.strip()
            else:
                # Detached HEAD, get commit SHA
                original_ref = run_cmd(
                    ["git", "rev-parse", "HEAD"],
                    cwd=repo_path,
                ).stdout.strip()
        except Exception as e:
            return {
                "instance_id": instance.instance_id,
                "status": "failed",
                "error": f"Failed to get current ref: {e}",
            }
        
        temp_commit_sha = None
        
        try:
            # Checkout to base commit
            checkout_result = run_cmd(
                ["git", "checkout", instance.base_sha],
                cwd=repo_path,
                check=False,
            )
            if checkout_result.returncode != 0:
                return {
                    "instance_id": instance.instance_id,
                    "status": "failed",
                    "error": f"Failed to checkout base commit: {checkout_result.stderr[:200]}",
                }
            
            # Reset any changes
            run_cmd(["git", "reset", "--hard"], cwd=repo_path, check=True)
            run_cmd(["git", "clean", "-fd"], cwd=repo_path, check=True)
            
            # Apply the fix patch
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False, encoding='utf-8') as tmp_patch:
                tmp_patch.write(instance.fix_patch)
                tmp_patch_path = Path(tmp_patch.name)
            
            try:
                # Try git apply first
                apply_result = run_cmd(
                    ["git", "apply", "--whitespace=nowarn", str(tmp_patch_path)],
                    cwd=repo_path,
                    check=False,
                )
                
                if apply_result.returncode != 0:
                    # Fallback to patch command
                    apply_result = run_cmd(
                        ["patch", "--batch", "--fuzz=5", "-p1", "-i", str(tmp_patch_path)],
                        cwd=repo_path,
                        check=False,
                    )
                    
                    if apply_result.returncode != 0:
                        return {
                            "instance_id": instance.instance_id,
                            "status": "failed",
                            "error": f"Failed to apply patch: {apply_result.stderr[:200]}",
                        }
            finally:
                tmp_patch_path.unlink(missing_ok=True)
            
            # Check if patch produced changes
            status_result = run_cmd(
                ["git", "status", "--porcelain"],
                cwd=repo_path,
            )
            if not status_result.stdout.strip():
                return {
                    "instance_id": instance.instance_id,
                    "status": "failed",
                    "error": "Patch produced no changes",
                }
            
            # Commit the changes
            run_cmd(["git", "add", "-A"], cwd=repo_path, check=True)
            commit_message = f"Golden patch for {instance.instance_id} (PR #{instance.pr_number})"
            commit_result = run_cmd(
                ["git", "commit", "-m", commit_message],
                cwd=repo_path,
                check=False,
            )
            
            if commit_result.returncode != 0:
                return {
                    "instance_id": instance.instance_id,
                    "status": "failed",
                    "error": f"Failed to commit changes: {commit_result.stderr[:200]}",
                }
            
            # Get the commit SHA
            temp_commit_sha = run_cmd(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
            ).stdout.strip()
            
            # Run RefactoringMiner on the commit
            rm_cmd = [
                str(self.refactoring_miner_path),
                "-c",
                str(repo_path),
                temp_commit_sha,
                "-json",
                str(output_json),
            ]
            
            rm_result = subprocess.run(
                rm_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=repo_path.parent,
            )
            
            if rm_result.returncode == 0 and output_json.exists():
                # Count refactorings (no filtering needed - fix_patch only contains source code changes)
                with open(output_json, 'r', encoding='utf-8') as f:
                    refactoring_data = json.load(f)
                
                total_refactorings = sum(
                    len(commit.get('refactorings', []))
                    for commit in refactoring_data.get('commits', [])
                )
                
                return {
                    "instance_id": instance.instance_id,
                    "status": "success",
                    "refactoring_count": total_refactorings,
                }
            else:
                error_msg = rm_result.stderr[:200] if rm_result.stderr else "Unknown error"
                return {
                    "instance_id": instance.instance_id,
                    "status": "failed",
                    "error": f"RefactoringMiner failed: {error_msg}",
                }
        
        except subprocess.TimeoutExpired:
            return {
                "instance_id": instance.instance_id,
                "status": "timeout",
                "error": f"Exceeded timeout of {self.timeout}s",
            }
        
        except Exception as e:
            return {
                "instance_id": instance.instance_id,
                "status": "failed",
                "error": str(e)[:200],
            }
        
        finally:
            # Cleanup: reset to original ref
            try:
                # First, reset any uncommitted changes
                run_cmd(["git", "reset", "--hard"], cwd=repo_path, check=False)
                run_cmd(["git", "clean", "-fd"], cwd=repo_path, check=False)
                
                # Checkout back to original ref
                run_cmd(["git", "checkout", original_ref], cwd=repo_path, check=False)
                
                # Reset again to ensure clean state
                run_cmd(["git", "reset", "--hard"], cwd=repo_path, check=False)
                run_cmd(["git", "clean", "-fd"], cwd=repo_path, check=False)
            except Exception as cleanup_error:
                logging.error(f"Cleanup failed for {instance.instance_id}: {cleanup_error}")
    
    def detect_all_instances(self, limit: Optional[int] = None) -> Dict[str, int]:
        """
        Detect refactorings for all golden instances (sequentially).
        
        Args:
            limit: Optional limit on number of instances to process (for testing)
        """
        logging.info("Starting golden refactoring detection")
        
        # Load instances
        instances = self.load_golden_instances()
        
        if not instances:
            logging.warning("No golden instances found")
            return {"total": 0, "success": 0, "failed": 0, "timeout": 0}
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            instances = instances[:limit]
            logging.info(f"Limited to first {limit} instances for testing")
        
        # Setup output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_path = self.output_dir / "detection.log"
        setup_logging(log_path)
        
        # Process instances sequentially
        stats = {
            "total": len(instances),
            "success": 0,
            "failed": 0,
            "timeout": 0,
            "total_refactorings": 0,
        }
        
        with tqdm(total=len(instances), desc="Detecting golden refactorings") as pbar:
            for instance in instances:
                output_json = self.output_dir / f"{instance.instance_id}.json"
                
                try:
                    result = self.detect_single_instance(instance, output_json)
                    instance_id = result["instance_id"]
                    status = result["status"]
                    
                    if status == "success":
                        stats["success"] += 1
                        refactoring_count = result.get("refactoring_count", 0)
                        stats["total_refactorings"] += refactoring_count
                        logging.info(f"✓ {instance_id} ({refactoring_count} refactorings)")
                    elif status == "timeout":
                        stats["timeout"] += 1
                        logging.warning(f"⏱ {instance_id}: {result.get('error', 'timeout')}")
                    else:
                        stats["failed"] += 1
                        logging.error(f"✗ {instance_id}: {result.get('error', 'unknown error')}")
                
                except Exception as e:
                    stats["failed"] += 1
                    logging.error(f"✗ {instance.instance_id}: Exception - {str(e)}")
                
                pbar.update(1)
        
        # Summary
        logging.info("Golden refactoring detection completed")
        logging.info(f"Total: {stats['total']}, Success: {stats['success']}, "
                    f"Failed: {stats['failed']}, Timeout: {stats['timeout']}")
        logging.info(f"Total refactorings found: {stats['total_refactorings']}")
        
        return stats


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Detect refactorings from golden patches (official PRs) using RefactoringMiner"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of instances to process (for testing, e.g., --limit 2)",
    )
    
    args = parser.parse_args()
    
    # Create detector instance
    detector = GoldenRefactoringDetector()
    
    print("=" * 80)
    print("Golden Refactoring Detector")
    print("=" * 80)
    print(f"Data directory:   {detector.golden_data_dir}")
    print(f"Output directory: {detector.output_dir}")
    if args.limit:
        print(f"Limit:            {args.limit} instances (testing mode)")
    print("=" * 80)
    
    # Detect refactorings
    stats = detector.detect_all_instances(limit=args.limit)
    
    print("\n" + "=" * 80)
    print("Detection Statistics:")
    print("=" * 80)
    print(f"Total:              {stats['total']}")
    print(f"Success:            {stats['success']}")
    print(f"Failed:             {stats['failed']}")
    print(f"Timeout:            {stats['timeout']}")
    if stats['total'] > 0:
        print(f"Success rate:       {stats['success']/stats['total']*100:.2f}%")
    print(f"Total refactorings: {stats['total_refactorings']}")
    print("=" * 80)


if __name__ == "__main__":
    main()