"""
Detect refactorings from agent patches using RefactoringMiner CLI.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

# Add project root to path for direct script execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.constant import (
    AGENT_REFACTORING_OUTPUT_DIR,
    PATCH_APPLY_OUTPUT_DIR,
    REFACTORING_DETECTION_TIMEOUT,
    REFACTORING_MINER_PATH,
    REPOS_DIR,
)


@dataclass
class CommitInfo:
    """Information about a commit to be analyzed."""
    instance_id: str
    agent: str
    repo: str
    commit: str


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


class AgentRefactoringDetector:
    """Detect refactorings from agent-generated patches."""
    
    def __init__(
        self,
        refactoring_miner_path: Path = REFACTORING_MINER_PATH,
        repos_dir: Path = REPOS_DIR,
        output_dir: Path = AGENT_REFACTORING_OUTPUT_DIR,
        timeout: int = REFACTORING_DETECTION_TIMEOUT,
        max_workers: int = 16,
    ):
        self.refactoring_miner_path = refactoring_miner_path
        self.repos_dir = repos_dir
        self.output_dir = output_dir
        self.timeout = timeout
        self.max_workers = max_workers
        
        if not self.refactoring_miner_path.exists():
            raise FileNotFoundError(f"RefactoringMiner not found at {self.refactoring_miner_path}")
    
    def load_agent_commit_mapping(self, agent_folder: str) -> List[CommitInfo]:
        """Load commit mapping for a specific agent folder."""
        mapping_file = PATCH_APPLY_OUTPUT_DIR / agent_folder / "agent_commit_mapping.json"
        
        if not mapping_file.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
        
        with open(mapping_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Filter only successful patches
        commits = []
        for item in data:
            if item.get("status") == "success":
                commits.append(
                    CommitInfo(
                        instance_id=item["instance_id"],
                        agent=item["agent"],
                        repo=item["repo"],
                        commit=item["commit"],
                    )
                )
        
        logging.info(f"Loaded {len(commits)} successful patches from {len(data)} total patches")
        return commits
    
    def detect_single_commit(self, commit_info: CommitInfo, output_json: Path) -> Dict:
        """Detect refactorings for a single commit using RefactoringMiner."""
        repo_path = self.repos_dir / commit_info.repo
        
        if not repo_path.exists():
            return {
                "instance_id": commit_info.instance_id,
                "status": "failed",
                "error": f"Repository not found: {repo_path}",
            }
        
        # Run RefactoringMiner
        cmd = [
            str(self.refactoring_miner_path),
            "-c",
            str(repo_path),
            commit_info.commit,
            "-json",
            str(output_json),
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=repo_path.parent,
            )
            
            if result.returncode == 0:
                # Verify output file was created
                if output_json.exists():
                    return {
                        "instance_id": commit_info.instance_id,
                        "status": "success",
                    }
                else:
                    return {
                        "instance_id": commit_info.instance_id,
                        "status": "failed",
                        "error": "Output file not created",
                    }
            else:
                return {
                    "instance_id": commit_info.instance_id,
                    "status": "failed",
                    "error": f"Exit code {result.returncode}: {result.stderr[:200]}",
                }
        
        except subprocess.TimeoutExpired:
            return {
                "instance_id": commit_info.instance_id,
                "status": "timeout",
                "error": f"Exceeded timeout of {self.timeout}s",
            }
        
        except Exception as e:
            return {
                "instance_id": commit_info.instance_id,
                "status": "failed",
                "error": str(e),
            }
    
    def _detect_commit_worker(self, args: tuple) -> Dict:
        """Worker function for parallel processing."""
        commit_info, output_json = args
        return self.detect_single_commit(commit_info, output_json)
    
    def detect_for_agent(self, agent_folder: str) -> Dict[str, int]:
        """Detect refactorings for all successful patches of an agent."""
        logging.info(f"Starting refactoring detection for agent: {agent_folder}")
        
        # Load commit mapping
        commits = self.load_agent_commit_mapping(agent_folder)
        
        if not commits:
            logging.warning(f"No successful patches found for {agent_folder}")
            return {"total": 0, "success": 0, "failed": 0, "timeout": 0}
        
        # Setup output directory
        agent_output_dir = self.output_dir / agent_folder
        agent_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging for this agent
        log_path = agent_output_dir / "detection.log"
        setup_logging(log_path)
        
        # Prepare tasks
        tasks = []
        for commit_info in commits:
            output_json = agent_output_dir / f"{commit_info.instance_id}.json"
            tasks.append((commit_info, output_json))
        
        # Process in parallel
        stats = {"total": len(commits), "success": 0, "failed": 0, "timeout": 0}
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._detect_commit_worker, task): task[0].instance_id
                for task in tasks
            }
            
            with tqdm(total=len(tasks), desc="Detecting refactorings") as pbar:
                for future in as_completed(futures):
                    instance_id = futures[future]
                    try:
                        result = future.result()
                        status = result["status"]
                        
                        if status == "success":
                            stats["success"] += 1
                            logging.info(f"✓ {instance_id}")
                        elif status == "timeout":
                            stats["timeout"] += 1
                            logging.warning(f"⏱ {instance_id}: {result.get('error', 'timeout')}")
                        else:
                            stats["failed"] += 1
                            logging.error(f"✗ {instance_id}: {result.get('error', 'unknown error')}")
                    
                    except Exception as e:
                        stats["failed"] += 1
                        logging.error(f"✗ {instance_id}: Exception in worker - {str(e)}")
                    
                    pbar.update(1)
        
        # Summary
        logging.info(f"Detection completed for {agent_folder}")
        logging.info(f"Total: {stats['total']}, Success: {stats['success']}, "
                    f"Failed: {stats['failed']}, Timeout: {stats['timeout']}")
        
        return stats
    
    def detect_all_agents(self) -> Dict[str, Dict[str, int]]:
        """Detect refactorings for all agents in patch_apply_results."""
        all_stats = {}
        
        # Find all agent folders with agent_commit_mapping.json
        agent_folders = []
        for folder in PATCH_APPLY_OUTPUT_DIR.iterdir():
            if folder.is_dir():
                mapping_file = folder / "agent_commit_mapping.json"
                if mapping_file.exists():
                    agent_folders.append(folder.name)
        
        logging.info(f"Found {len(agent_folders)} agent folders to process")
        
        for agent_folder in agent_folders:
            try:
                stats = self.detect_for_agent(agent_folder)
                all_stats[agent_folder] = stats
            except Exception as e:
                logging.error(f"Failed to process {agent_folder}: {e}")
                all_stats[agent_folder] = {"error": str(e)}
        
        return all_stats


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Detect refactorings from agent-generated patches using RefactoringMiner"
    )
    parser.add_argument(
        "--agent",
        type=str,
        help="Detect refactorings for a specific agent folder (e.g., MagentLess_Claude-3.5-Sonnet(Oct))",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers (default: 16)",
    )
    
    args = parser.parse_args()
    
    # Create detector instance
    detector = AgentRefactoringDetector(max_workers=args.workers)
    
    print("=" * 80)
    print(f"Agent Refactoring Detector (Workers: {detector.max_workers})")
    print("=" * 80)
    
    # Detect for specific agent or all agents
    if args.agent:
        print(f"Detecting refactorings for agent: {args.agent}")
        print("=" * 80)
        stats = detector.detect_for_agent(args.agent)
        print("\n" + "=" * 80)
        print("Detection Statistics:")
        print("=" * 80)
        print(f"Total:    {stats['total']}")
        print(f"Success:  {stats['success']}")
        print(f"Failed:   {stats['failed']}")
        print(f"Timeout:  {stats['timeout']}")
        if stats['total'] > 0:
            print(f"Success rate: {stats['success']/stats['total']*100:.2f}%")
        print("=" * 80)
    else:
        print("Detecting refactorings for ALL agents")
        print("=" * 80)
        all_stats = detector.detect_all_agents()
        
        # Summary
        print("\n" + "=" * 80)
        print("Overall Detection Statistics:")
        print("=" * 80)
        
        total_all = sum(s.get('total', 0) for s in all_stats.values() if 'total' in s)
        success_all = sum(s.get('success', 0) for s in all_stats.values() if 'success' in s)
        failed_all = sum(s.get('failed', 0) for s in all_stats.values() if 'failed' in s)
        timeout_all = sum(s.get('timeout', 0) for s in all_stats.values() if 'timeout' in s)
        
        print(f"Agents processed:  {len(all_stats)}")
        print(f"Total commits:     {total_all}")
        print(f"Success:           {success_all}")
        print(f"Failed:            {failed_all}")
        print(f"Timeout:           {timeout_all}")
        if total_all > 0:
            print(f"Overall success rate: {success_all/total_all*100:.2f}%")
        print("=" * 80)


if __name__ == "__main__":
    main()
