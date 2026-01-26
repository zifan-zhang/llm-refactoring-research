"""
Apply the agents generated patches and create the commit mapping.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from src.constant import (
    JAVA_EXPERIMENT_DIR,
    MSB_JAVA_DIR,
    PATCH_APPLY_OUTPUT_DIR,
    REPOS_DIR,
    TAG_PREFIX,
)


@dataclass(frozen=True)
class InstanceInfo:
    instance_id: str
    base_sha: str


@dataclass(frozen=True)
class AgentPatch:
    instance_id: str
    agent_name: str
    patch_text: str


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def run_cmd(
    args: List[str],
    cwd: Path,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=str(cwd),
        check=check,
        text=True,
        capture_output=capture_output,
    )


def git_status_clean(repo_dir: Path) -> bool:
    result = run_cmd(["git", "status", "--porcelain"], cwd=repo_dir, check=True)
    return result.stdout.strip() == ""


def get_current_ref(repo_dir: Path) -> str:
    result = run_cmd(
        ["git", "symbolic-ref", "-q", "--short", "HEAD"],
        cwd=repo_dir,
        check=False,
    )
    ref = result.stdout.strip()
    if ref:
        return ref
    result = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True)
    return result.stdout.strip()


def checkout_ref(repo_dir: Path, ref: str) -> bool:
    result = run_cmd(["git", "checkout", ref], cwd=repo_dir, check=False)
    if result.returncode != 0:
        logging.error("Checkout failed for %s: %s", ref, result.stderr.strip())
        return False
    return True


def sanitize_tag(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return sanitized.strip("_")


def ensure_unique_tag(repo_dir: Path, tag_name: str) -> str:
    candidate = tag_name
    suffix = 1
    while True:
        result = run_cmd(
            ["git", "tag", "--list", candidate], cwd=repo_dir, check=True
        )
        if not result.stdout.strip():
            return candidate
        candidate = f"{tag_name}-{suffix}"
        suffix += 1


def ensure_trailing_newline(path: Path) -> None:
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        path.write_bytes(data + b"\n")


def load_instance_info(java_dir: Path) -> Dict[str, InstanceInfo]:
    instance_map: Dict[str, InstanceInfo] = {}
    for jsonl_path in sorted(java_dir.glob("*.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                instance_id = payload.get("instance_id")
                base = payload.get("base", {})
                base_sha = base.get("sha")
                if not instance_id or not base_sha:
                    continue
                instance_map[instance_id] = InstanceInfo(
                    instance_id=instance_id,
                    base_sha=base_sha,
                )
    return instance_map


def load_agent_patches(preds_path: Path) -> List[AgentPatch]:
    patches: List[AgentPatch] = []
    with preds_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            instance_id = payload.get("instance_id")
            patch_text = payload.get("model_patch", "")
            agent_name = payload.get("model_name_or_path") or "unknown-agent"
            if not instance_id:
                continue
            # Include empty patches - they will be handled in process_instance
            patches.append(
                AgentPatch(
                    instance_id=instance_id,
                    agent_name=agent_name,
                    patch_text=patch_text,
                )
            )
    return patches


def parse_repo_name(instance_id: str) -> Optional[str]:
    if "__" not in instance_id:
        return None
    repo_part = instance_id.split("__", 1)[1]
    if "-" not in repo_part:
        return None
    repo_name, _ = repo_part.rsplit("-", 1)
    return repo_name


def write_patch_file(
    work_dir: Path, instance_id: str, fix_patch: str
) -> Path:
    instance_dir = work_dir / sanitize_tag(instance_id)
    instance_dir.mkdir(parents=True, exist_ok=True)
    fix_patch_path = instance_dir / "fix.patch"
    fix_patch_path.write_text(fix_patch, encoding="utf-8")
    return fix_patch_path


def git_apply_check(repo_dir: Path, patch_path: Path) -> Tuple[bool, str]:
    args = ["git", "apply", "--whitespace=nowarn", "--check", str(patch_path)]
    result = run_cmd(args, cwd=repo_dir, check=False)
    if result.returncode == 0:
        return True, ""
    return False, result.stderr.strip()


def git_apply(repo_dir: Path, patch_path: Path) -> Tuple[bool, str]:
    args = ["git", "apply", "--whitespace=nowarn", str(patch_path)]
    result = run_cmd(args, cwd=repo_dir, check=False)
    if result.returncode == 0:
        return True, ""
    return False, result.stderr.strip()


def patch_apply_check(repo_dir: Path, patch_path: Path) -> Tuple[bool, str]:
    args = [
        "patch",
        "--batch",
        "--fuzz=5",
        "-p1",
        "--dry-run",
        "-i",
        str(patch_path),
    ]
    result = run_cmd(args, cwd=repo_dir, check=False)
    if result.returncode == 0:
        return True, ""
    return False, result.stderr.strip()


def patch_apply(repo_dir: Path, patch_path: Path) -> Tuple[bool, str]:
    args = [
        "patch",
        "--batch",
        "--fuzz=5",
        "-p1",
        "-i",
        str(patch_path),
    ]
    result = run_cmd(args, cwd=repo_dir, check=False)
    if result.returncode == 0:
        return True, ""
    return False, result.stderr.strip()


def apply_patch_with_strategy(
    repo_dir: Path, fix_patch_path: Path
) -> Tuple[bool, str]:
    # Strategy 1: Try git apply first
    ok, error = git_apply_check(repo_dir, fix_patch_path)
    if ok:
        ok, error = git_apply(repo_dir, fix_patch_path)
        if ok:
            return True, ""
        return False, error

    # Strategy 2: Fallback to patch command with fuzz
    ensure_trailing_newline(fix_patch_path)
    ok, error = patch_apply_check(repo_dir, fix_patch_path)
    if not ok:
        return False, error

    ok, error = patch_apply(repo_dir, fix_patch_path)
    if not ok:
        return False, error
    return True, ""


def commit_and_tag(
    repo_dir: Path, instance_id: str, agent_name: str, agent_dir_name: str
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not git_status_clean(repo_dir):
        run_cmd(["git", "add", "-A"], cwd=repo_dir, check=True)
        commit_message = f"Apply agent patch for {agent_dir_name}/{instance_id}"
        commit_result = run_cmd(
            ["git", "commit", "-m", commit_message],
            cwd=repo_dir,
            check=False,
        )
        if commit_result.returncode != 0:
            return None, None, commit_result.stderr.strip()

    commit_sha = run_cmd(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True
    ).stdout.strip()

    # Include agent_dir_name in tag to avoid conflicts between different agents
    tag_base = f"{TAG_PREFIX}_{sanitize_tag(agent_dir_name)}_{sanitize_tag(agent_name)}_{sanitize_tag(instance_id)}"
    tag_name = ensure_unique_tag(repo_dir, tag_base)
    tag_result = run_cmd(["git", "tag", tag_name], cwd=repo_dir, check=False)
    if tag_result.returncode != 0:
        return commit_sha, None, tag_result.stderr.strip()
    return commit_sha, tag_name, None


def process_instance(
    repo_dir: Path,
    original_ref: str,
    instance: AgentPatch,
    instance_info: InstanceInfo,
    patch_work_dir: Path,
    agent_dir_name: str,
) -> Dict[str, str]:
    result: Dict[str, str] = {
        "instance_id": instance.instance_id,
        "agent": instance.agent_name,
        "repo": repo_dir.name,
        "base_sha": instance_info.base_sha,
        "status": "failed",
    }

    # Check for empty patch
    if not instance.patch_text or not instance.patch_text.strip():
        result["error"] = "empty patch"
        result["status"] = "skipped"
        return result

    if not git_status_clean(repo_dir):
        result["error"] = "dirty worktree"
        return result

    if not checkout_ref(repo_dir, instance_info.base_sha):
        result["error"] = "checkout base failed"
        return result

    try:
        if not git_status_clean(repo_dir):
            result["error"] = "dirty worktree after checkout"
            return result

        fix_patch_path = write_patch_file(
            patch_work_dir,
            instance.instance_id,
            instance.patch_text,
        )

        ok, error = apply_patch_with_strategy(repo_dir, fix_patch_path)
        if not ok:
            result["error"] = f"apply failed: {error}"
            return result

        if git_status_clean(repo_dir):
            result["error"] = "patch produced no changes"
            return result

        commit_sha, tag_name, error = commit_and_tag(
            repo_dir, instance.instance_id, instance.agent_name, agent_dir_name
        )
        if error:
            result["error"] = error
            if commit_sha:
                result["commit"] = commit_sha
            return result

        result.update(
            {
                "status": "success",
                "commit": commit_sha or "",
                "tag": tag_name or "",
            }
        )
        return result
    finally:
        checkout_ref(repo_dir, original_ref)


def save_mapping(mapping: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=2, ensure_ascii=False)


def compute_and_save_statistics(results: List[Dict[str, str]], stats_path: Path) -> None:
    from collections import defaultdict

    total = len(results)
    success_count = sum(1 for r in results if r["status"] == "success")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    failed_count = sum(1 for r in results if r["status"] == "failed")

    # Count by repo
    repo_stats = defaultdict(lambda: {"total": 0, "success": 0, "skipped": 0, "failed": 0})
    for result in results:
        repo = result.get("repo", "unknown")
        repo_stats[repo]["total"] += 1
        if result["status"] == "success":
            repo_stats[repo]["success"] += 1
        elif result["status"] == "skipped":
            repo_stats[repo]["skipped"] += 1
        else:
            repo_stats[repo]["failed"] += 1

    # Count failure/skip reasons
    failure_reasons = defaultdict(int)
    for result in results:
        if result["status"] == "failed" or result["status"] == "skipped":
            error = result.get("error", "unknown error")
            # Simplify error messages
            if "empty patch" in error:
                failure_reasons["empty patch"] += 1
            elif "apply failed" in error:
                failure_reasons["apply failed"] += 1
            elif "checkout" in error:
                failure_reasons["checkout failed"] += 1
            elif "dirty worktree" in error:
                failure_reasons["dirty worktree"] += 1
            elif "repo not found" in error:
                failure_reasons["repo not found"] += 1
            elif "base info missing" in error:
                failure_reasons["base info missing"] += 1
            else:
                failure_reasons["other"] += 1

    statistics = {
        "summary": {
            "total": total,
            "success": success_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "success_rate": f"{success_count / total * 100:.2f}%" if total > 0 else "0%",
        },
        "by_repo": dict(repo_stats),
        "failure_reasons": dict(failure_reasons),
    }

    # Save to JSON
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)

    # Print to console
    print("\n" + "=" * 60)
    print("PATCH APPLICATION STATISTICS")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Total patches: {total}")
    print(f"  ✅ Success: {success_count}")
    print(f"  ⊘ Skipped (empty): {skipped_count}")
    print(f"  ❌ Failed: {failed_count}")
    print(f"  Success rate: {statistics['summary']['success_rate']}")

    print(f"\nBy Repository:")
    for repo, stats in sorted(repo_stats.items()):
        rate = (
            f"{stats['success'] / stats['total'] * 100:.1f}%"
            if stats["total"] > 0
            else "0%"
        )
        print(
            f"  {repo:25s}: {stats['success']:3d}/{stats['total']:3d} ({rate}) [skipped: {stats['skipped']}]"
        )

    if failure_reasons:
        print(f"\nFailure/Skip Reasons:")
        for reason, count in sorted(
            failure_reasons.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {reason:25s}: {count}")

    print(f"\n📁 Results saved to: {stats_path.parent}")
    print("=" * 60 + "\n")


def process_agent(
    agent_dir: Path,
    agent_output_dir: Path,
    instance_info_map: Dict[str, InstanceInfo],
) -> None:
    agent_name = agent_dir.name
    preds_path = agent_dir / "all_preds.jsonl"
    
    if not preds_path.exists():
        print(f"⚠️  Skipping {agent_name}: all_preds.jsonl not found")
        return

    # Setup paths for this agent
    log_path = agent_output_dir / "patch_apply.log"
    commit_mapping_path = agent_output_dir / "agent_commit_mapping.json"
    statistics_path = agent_output_dir / "statistics.json"
    patch_work_dir = agent_output_dir / "patch_work"

    # Setup logging for this agent
    setup_logging(log_path)

    # Load patches
    agent_patches = load_agent_patches(preds_path)
    
    # Group by repo
    grouped: Dict[str, List[AgentPatch]] = defaultdict(list)
    for patch in agent_patches:
        repo_name = parse_repo_name(patch.instance_id)
        if repo_name is None:
            logging.error("Invalid instance_id: %s", patch.instance_id)
            continue
        grouped[repo_name].append(patch)

    # Print initial information
    print("\n" + "=" * 80)
    print(f"Processing Agent: {agent_name}")
    print("=" * 80)
    print(f"Total patches to process: {len(agent_patches)}")
    print(f"Repositories: {len(grouped)}")
    for repo_name, patches in sorted(grouped.items()):
        print(f"  - {repo_name}: {len(patches)} patches")
    print("=" * 80 + "\n")

    results: List[Dict[str, str]] = []
    patch_work_dir.mkdir(parents=True, exist_ok=True)

    for repo_name, patches in grouped.items():
        repo_dir = REPOS_DIR / repo_name
        if not repo_dir.exists():
            for patch in patches:
                logging.error("Repo not found for %s: %s", patch.instance_id, repo_dir)
                results.append(
                    {
                        "instance_id": patch.instance_id,
                        "agent": patch.agent_name,
                        "repo": repo_name,
                        "status": "failed",
                        "error": "repo not found",
                    }
                )
            continue

        original_ref = get_current_ref(repo_dir)

        for patch in tqdm(patches, desc=f"{repo_name}", unit="instance"):
            instance_info = instance_info_map.get(patch.instance_id)
            if instance_info is None:
                logging.error("Base info missing for %s", patch.instance_id)
                results.append(
                    {
                        "instance_id": patch.instance_id,
                        "agent": patch.agent_name,
                        "repo": repo_name,
                        "status": "failed",
                        "error": "base info missing",
                    }
                )
                continue

            result = process_instance(
                repo_dir, original_ref, patch, instance_info, patch_work_dir, agent_name
            )
            results.append(result)
            if result["status"] == "success":
                logging.info(
                    "Applied patch for %s -> %s",
                    patch.instance_id,
                    result.get("commit"),
                )
            elif result["status"] == "skipped":
                logging.warning(
                    "Skipped patch for %s: %s",
                    patch.instance_id,
                    result.get("error", "unknown reason"),
                )
            else:
                logging.error(
                    "Failed patch for %s: %s",
                    patch.instance_id,
                    result.get("error", "unknown error"),
                )

    save_mapping(results, commit_mapping_path)
    compute_and_save_statistics(results, statistics_path)


def main() -> None:
    # Load instance info once (shared across all agents)
    print("\n" + "=" * 80)
    print("BATCH PATCH APPLICATION FOR ALL AGENTS")
    print("=" * 80)
    print("Loading instance information from Multi-SWE-bench...")
    instance_info_map = load_instance_info(MSB_JAVA_DIR)
    print(f"✅ Loaded {len(instance_info_map)} instances\n")

    # Get all agent directories
    agent_dirs = sorted([d for d in JAVA_EXPERIMENT_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(agent_dirs)} agents to process\n")

    # Process each agent
    for i, agent_dir in enumerate(agent_dirs, 1):
        agent_name = agent_dir.name
        agent_output_dir = PATCH_APPLY_OUTPUT_DIR / agent_name
        
        print(f"\n[{i}/{len(agent_dirs)}] Starting agent: {agent_name}")
        
        try:
            process_agent(agent_dir, agent_output_dir, instance_info_map)
        except Exception as e:
            print(f"❌ Error processing {agent_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("ALL AGENTS PROCESSED")
    print("=" * 80)
    print(f"Results saved to: {PATCH_APPLY_OUTPUT_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()



