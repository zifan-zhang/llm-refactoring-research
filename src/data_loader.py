"""
Data loader for unified data builder
"""

import json
import jsonlines
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from src.constant import (
    DATA_DIR,
    MSB_JAVA_DIR,
    PATCH_APPLY_OUTPUT_DIR,
    REFACTORING_DETECTION_OUTPUT_DIR,
    EVALUATION_LOGS_DIR
)


class TaskDifficultyLoader:
    """Loader for task difficulty from index.json"""
    
    def __init__(self, index_path: Optional[str] = None):
        """
        Initialize loader
        
        Args:
            index_path: Path to index.json, defaults to data/index.json
        """
        if index_path is None:
            index_path = DATA_DIR / "index.json"
        
        self.index_path = Path(index_path)
        self._difficulty_map = None
    
    def load(self) -> Dict[str, str]:
        """
        Load difficulty mapping
        
        Returns:
            Dictionary mapping instance_id to difficulty (Easy/Medium/Hard)
        """
        if self._difficulty_map is not None:
            return self._difficulty_map
        
        with open(self.index_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        difficulty_map = {}
        
        # Map each ID to its difficulty level
        for id_ in data.get('easy_ids', []):
            difficulty_map[id_] = 'Easy'
        
        for id_ in data.get('medium_ids', []):
            difficulty_map[id_] = 'Medium'
        
        for id_ in data.get('hard_ids', []):
            difficulty_map[id_] = 'Hard'
        
        self._difficulty_map = difficulty_map
        return difficulty_map
    
    def get_difficulty(self, instance_id: str) -> str:
        """
        Get difficulty for a specific instance
        
        Args:
            instance_id: Instance ID
            
        Returns:
            Difficulty level or 'Unknown' if not found
        """
        if self._difficulty_map is None:
            self.load()
        
        return self._difficulty_map.get(instance_id, 'Unknown')


class IssueTypeLoader:
    """Loader for issue types from Excel file"""
    
    def __init__(self, excel_path: Optional[str] = None):
        """
        Initialize loader
        
        Args:
            excel_path: Path to issue_types.xlsx
        """
        if excel_path is None:
            excel_path = DATA_DIR / "issue_types.xlsx"
        
        self.excel_path = Path(excel_path)
        self._issue_map = None
    
    def load(self) -> Dict[str, str]:
        """
        Load issue type mapping
        
        Returns:
            Dictionary mapping instance_id to issue_type
        """
        if self._issue_map is not None:
            return self._issue_map
        
        df = pd.read_excel(self.excel_path)
        
        # Get the key columns
        id_col = '示例id(instance_id)'
        tag_col = '标签'  # Column 4: the actual issue type label
        
        issue_map = {}
        
        for _, row in df.iterrows():
            instance_id = row.get(id_col)
            tag = row.get(tag_col, '')
            
            if pd.isna(instance_id):
                continue
            
            # Extract instance_id (remove prefix if exists)
            # Format: "50_java_apache-dubbo-10638" -> "apache-dubbo-10638"
            # or "apache__dubbo-10638"
            instance_id_str = str(instance_id)
            if '_java_' in instance_id_str:
                parts = instance_id_str.split('_java_')
                if len(parts) > 1:
                    instance_id_clean = parts[1].replace('-', '__', 1)
                else:
                    instance_id_clean = instance_id_str
            else:
                instance_id_clean = instance_id_str
            
            # Use the tag column directly and normalize it
            issue_type = self._normalize_issue_type(tag)
            issue_map[instance_id_clean] = issue_type
        
        self._issue_map = issue_map
        return issue_map
    
    def _normalize_issue_type(self, tag: str) -> str:
        """
        Normalize issue type from tag column
        
        Args:
            tag: Issue type tag from Excel
            
        Returns:
            Normalized issue type category
        """
        if pd.isna(tag):
            return 'unknown'
        
        tag_lower = str(tag).strip().lower()
        
        # Normalize the tag values from column 4
        if tag_lower == 'bug fix':
            return 'bug_fix'
        elif tag_lower == 'new feature':
            return 'new_feature'
        elif tag_lower == 'feature optimization':
            return 'feature_optimization'
        else:
            return 'unknown'
    
    def get_issue_type(self, instance_id: str) -> str:
        """
        Get issue type for a specific instance
        
        Args:
            instance_id: Instance ID
            
        Returns:
            Issue type or 'unknown' if not found
        """
        if self._issue_map is None:
            self.load()
        
        return self._issue_map.get(instance_id, 'unknown')


class RefactoringDataLoader:
    """Loader for refactoring detection results"""
    
    def __init__(self, refactoring_base_dir: Optional[str] = None):
        """
        Initialize loader
        
        Args:
            refactoring_base_dir: Base directory for refactoring results
        """
        if refactoring_base_dir is None:
            refactoring_base_dir = REFACTORING_DETECTION_OUTPUT_DIR
        
        self.base_dir = Path(refactoring_base_dir)
        self.agent_dir = self.base_dir / "agent"
        self.golden_dir = self.base_dir / "golden"
    
    def get_agent_refactoring_path(self, agent_name: str, instance_id: str) -> Path:
        """
        Get path to agent refactoring JSON file
        
        Args:
            agent_name: Name of the agent
            instance_id: Instance ID
            
        Returns:
            Path to refactoring JSON file
        """
        return self.agent_dir / agent_name / f"{instance_id}.json"
    
    def get_golden_refactoring_path(self, instance_id: str) -> Path:
        """
        Get path to golden refactoring JSON file
        
        Args:
            instance_id: Instance ID
            
        Returns:
            Path to refactoring JSON file
        """
        return self.golden_dir / f"{instance_id}.json"
    
    def load_agent_refactoring_data(self, agent_name: str, instance_id: str) -> Optional[Dict]:
        """
        Load agent refactoring data from JSON file
        
        Args:
            agent_name: Name of the agent
            instance_id: Instance ID
            
        Returns:
            Refactoring data dictionary or None if not found
        """
        ref_path = self.get_agent_refactoring_path(agent_name, instance_id)
        if not ref_path.exists():
            return None
        
        try:
            with open(ref_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    def format_refactoring_context(self, agent_name: str, instance_id: str) -> str:
        """
        Format refactoring data as context string for LLM prompt
        
        Args:
            agent_name: Name of the agent
            instance_id: Instance ID
            
        Returns:
            Formatted refactoring context string
        """
        ref_data = self.load_agent_refactoring_data(agent_name, instance_id)
        
        if not ref_data or 'commits' not in ref_data:
            return "[no refactoring context available]"
        
        lines = []
        lines.append("The following refactorings were detected in the patch:\n")
        
        for commit in ref_data['commits']:
            refactorings = commit.get('refactorings', [])
            
            for i, ref in enumerate(refactorings, 1):
                ref_type = ref.get('type', 'Unknown')
                description = ref.get('description', '')
                
                lines.append(f"{i}. {ref_type}")
                lines.append(f"   Description: {description}")
                
                # Format left side locations
                left_locations = ref.get('leftSideLocations', [])
                if left_locations:
                    lines.append("   Before:")
                    for loc in left_locations:
                        file_path = loc.get('filePath', '')
                        start_line = loc.get('startLine', '')
                        end_line = loc.get('endLine', '')
                        code_element = loc.get('codeElement', '')
                        loc_desc = loc.get('description', '')
                        
                        lines.append(f"     - File: {file_path}")
                        lines.append(f"       Lines: {start_line}-{end_line}")
                        if code_element:
                            lines.append(f"       Element: {code_element}")
                        if loc_desc:
                            lines.append(f"       Description: {loc_desc}")
                
                # Format right side locations
                right_locations = ref.get('rightSideLocations', [])
                if right_locations:
                    lines.append("   After:")
                    for loc in right_locations:
                        file_path = loc.get('filePath', '')
                        start_line = loc.get('startLine', '')
                        end_line = loc.get('endLine', '')
                        code_element = loc.get('codeElement', '')
                        loc_desc = loc.get('description', '')
                        
                        lines.append(f"     - File: {file_path}")
                        lines.append(f"       Lines: {start_line}-{end_line}")
                        if code_element:
                            lines.append(f"       Element: {code_element}")
                        if loc_desc:
                            lines.append(f"       Description: {loc_desc}")
                
                lines.append("")  # Empty line between refactorings
        
        return "\n".join(lines)


class PatchDataLoader:
    """Loader for patch files and related data"""
    
    def __init__(self, patch_base_dir: Optional[str] = None):
        """
        Initialize loader
        
        Args:
            patch_base_dir: Base directory for patch results
        """
        if patch_base_dir is None:
            patch_base_dir = PATCH_APPLY_OUTPUT_DIR
        
        self.base_dir = Path(patch_base_dir)
    
    def get_patch_path(self, agent_name: str, instance_id: str) -> Path:
        """
        Get path to agent patch file
        
        Args:
            agent_name: Name of the agent
            instance_id: Instance ID
            
        Returns:
            Path to fix.patch file
        """
        return self.base_dir / agent_name / "patch_work" / instance_id / "fix.patch"
    
    def get_agent_patch_content(self, agent_name: str, instance_id: str) -> str:
        """
        Get agent patch content as string
        
        Args:
            agent_name: Name of the agent
            instance_id: Instance ID
            
        Returns:
            Patch content string or empty string if not found
        """
        patch_path = self.get_patch_path(agent_name, instance_id)
        if not patch_path.exists():
            return ""
        
        with open(patch_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def get_modified_lines(self, agent_name: str, instance_id: str) -> int:
        """
        Get agent patch modified lines count (added + deleted).
        
        Args:
            agent_name: Name of the agent
            instance_id: Instance ID
            
        Returns:
            Total number of lines changed (added + deleted)
        """
        from src.utils.patch_utils import get_patch_size
        patch_path = self.get_patch_path(agent_name, instance_id)
        return get_patch_size(str(patch_path))
    
    def get_modified_files(self, agent_name: str, instance_id: str) -> int:
        """
        Get agent patch modified files count.
        
        Args:
            agent_name: Name of the agent
            instance_id: Instance ID
            
        Returns:
            Number of modified files
        """
        from src.utils.patch_utils import get_modified_files_from_patch
        patch_path = self.get_patch_path(agent_name, instance_id)
        return get_modified_files_from_patch(str(patch_path))


class GoldenPatchLoader:
    """Loader for golden patches from Multi-SWE-bench JSONL files"""
    
    def __init__(self, msb_java_dir: Optional[str] = None):
        """
        Initialize loader
        
        Args:
            msb_java_dir: Directory containing Multi-SWE-bench Java JSONL files
        """
        if msb_java_dir is None:
            msb_java_dir = MSB_JAVA_DIR
        
        self.base_dir = Path(msb_java_dir)
        self._cache = {}
    
    def _load_jsonl_file(self, file_path: Path) -> Dict[str, Dict]:
        """
        Load JSONL file and cache by instance_id
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            Dictionary mapping instance_id to record
        """
        cache = {}
        
        with jsonlines.open(file_path) as reader:
            for obj in reader:
                org = obj.get('org', '')
                repo = obj.get('repo', '')
                number = obj.get('number', '')
                instance_id = f"{org}__{repo}-{number}"
                cache[instance_id] = obj
        
        return cache
    
    def _ensure_loaded(self):
        """Load all JSONL files if not already loaded"""
        if self._cache:
            return
        
        # Load all JSONL files in the directory
        for jsonl_file in self.base_dir.glob("*.jsonl"):
            file_cache = self._load_jsonl_file(jsonl_file)
            self._cache.update(file_cache)
    
    def get_golden_patch(self, instance_id: str) -> str:
        """
        Get golden patch content for an instance
        
        Args:
            instance_id: Instance ID (format: org__repo-number)
            
        Returns:
            Golden patch content or empty string if not found
        """
        self._ensure_loaded()
        
        record = self._cache.get(instance_id)
        if record:
            return record.get('fix_patch', '')
        
        return ""
    
    def get_golden_record(self, instance_id: str) -> Optional[Dict]:
        """
        Get complete golden record for an instance
        
        Args:
            instance_id: Instance ID
            
        Returns:
            Golden record dictionary or None if not found
        """
        self._ensure_loaded()
        return self._cache.get(instance_id)
    
    def get_issue_description(self, instance_id: str) -> str:
        """
        Get formatted issue description for an instance
        
        Args:
            instance_id: Instance ID
            
        Returns:
            Formatted issue description string
        """
        record = self.get_golden_record(instance_id)
        
        if not record:
            return "[no issue description available]"
        
        resolved_issues = record.get('resolved_issues', [])
        
        if not resolved_issues:
            return "[no issue description available]"
        
        # Format all resolved issues
        lines = []
        for i, issue in enumerate(resolved_issues, 1):
            issue_number = issue.get('number', 'N/A')
            issue_title = issue.get('title', 'N/A')
            issue_body = issue.get('body', '')
            
            if len(resolved_issues) > 1:
                lines.append(f"Issue #{issue_number}: {issue_title}")
            else:
                lines.append(f"Issue #{issue_number}: {issue_title}")
            
            if issue_body:
                lines.append(issue_body)
            
            if i < len(resolved_issues):
                lines.append("\n---\n")
        
        return "\n".join(lines)


class CompilationLogLoader:
    """Loader for compilation logs and analysis"""
    
    def __init__(self, eval_logs_base_dir: Optional[str] = None):
        """
        Initialize loader
        
        Args:
            eval_logs_base_dir: Base directory for evaluation logs
        """
        if eval_logs_base_dir is None:
            eval_logs_base_dir = EVALUATION_LOGS_DIR
        
        self.base_dir = Path(eval_logs_base_dir)
    
    def get_compilation_log_path(self, agent_name: str, instance_id: str) -> Path:
        """
        Get path to compilation log file
        
        Args:
            agent_name: Name of the agent
            instance_id: Instance ID (format: org__repo-number)
            
        Returns:
            Path to fix-patch-run.log file
        """
        # Parse instance_id to get org, repo, and pr number
        # Format: org__repo-pr-number or org/repo:pr-number
        if '__' in instance_id:
            parts = instance_id.split('__')
            org = parts[0]
            repo_pr = parts[1]
            
            if '-pr-' in repo_pr:
                repo, pr_num = repo_pr.rsplit('-pr-', 1)
                pr_id = f"pr-{pr_num}"
            else:
                # Try to extract pr number from the end
                pr_parts = repo_pr.rsplit('-', 1)
                if len(pr_parts) == 2:
                    repo, pr_num = pr_parts
                    pr_id = f"pr-{pr_num}"
                else:
                    repo = repo_pr
                    pr_id = ""
        else:
            # Handle alternative format
            parts = instance_id.split(':')
            if len(parts) == 2:
                org_repo = parts[0]
                pr_id = parts[1]
                org, repo = org_repo.split('/', 1)
            else:
                return Path("")
        
        log_path = self.base_dir / agent_name / "workdir" / org / repo / "evals" / pr_id / "fix-patch-run.log"
        return log_path
    
    def get_compilation_result(self, agent_name: str, instance_id: str) -> bool:
        """
        Get compilation result for an agent and instance
        
        Args:
            agent_name: Name of the agent
            instance_id: Instance ID
            
        Returns:
            True if compilation succeeded, False otherwise
        """
        from src.utils.compilation_utils import analyze_compilation_log
        
        log_path = self.get_compilation_log_path(agent_name, instance_id)
        
        if not log_path.exists():
            # If log file doesn't exist, assume compilation failed
            return False
        
        try:
            result = analyze_compilation_log(log_path, instance_id)
            return result.is_compile_ok
        except Exception:
            # If analysis fails, assume compilation failed
            return False


class FinalReportLoader:
    """Loader for final_report.json files"""
    
    def __init__(self, eval_logs_base_dir: Optional[str] = None):
        """
        Initialize loader
        
        Args:
            eval_logs_base_dir: Base directory for evaluation logs
        """
        if eval_logs_base_dir is None:
            eval_logs_base_dir = EVALUATION_LOGS_DIR
        
        self.base_dir = Path(eval_logs_base_dir)
        self._cache = {}
    
    def _load_final_report(self, agent_name: str) -> Dict:
        """
        Load final_report.json for an agent
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Final report dictionary
        """
        if agent_name in self._cache:
            return self._cache[agent_name]
        
        report_path = self.base_dir / agent_name / "output" / "final_report.json"
        
        if not report_path.exists():
            self._cache[agent_name] = {}
            return {}
        
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            self._cache[agent_name] = report
            return report
        except Exception:
            self._cache[agent_name] = {}
            return {}
    
    def get_issue_status(self, agent_name: str, instance_id: str) -> str:
        """
        Get issue resolution status
        
        Args:
            agent_name: Name of the agent
            instance_id: Instance ID (format: org__repo-number)
            
        Returns:
            "resolved" if issue is resolved, "unresolved" otherwise
        """
        report = self._load_final_report(agent_name)
        
        resolved_ids = report.get('resolved_ids', [])
        unresolved_ids = report.get('unresolved_ids', [])
        
        # Convert instance_id format if needed
        # From org__repo-number to org/repo:pr-number
        # Example: elastic__logstash-13930 -> elastic/logstash:pr-13930
        search_id = instance_id
        if '__' in instance_id:
            parts = instance_id.split('__')
            org = parts[0]
            repo_pr = parts[1]
            
            # Split repo and pr number using rsplit to handle repo names with dashes
            # Example: "logstash-13930" -> "logstash", "13930"
            pr_parts = repo_pr.rsplit('-', 1)
            if len(pr_parts) == 2:
                repo, pr_num = pr_parts
                search_id = f"{org}/{repo}:pr-{pr_num}"
            else:
                search_id = f"{org}/{repo_pr}"
        
        # Check both formats
        if instance_id in resolved_ids or search_id in resolved_ids:
            return "resolved"
        elif instance_id in unresolved_ids or search_id in unresolved_ids:
            return "unresolved"
        else:
            # Default to unresolved if not found in either list
            return "unresolved"


class PatchApplyResultsLoader:
    """Loader for patch apply results (agent_commit_mapping.json)"""
    
    def __init__(self, patch_results_base_dir: Optional[str] = None):
        """
        Initialize loader
        
        Args:
            patch_results_base_dir: Base directory for patch apply results
        """
        if patch_results_base_dir is None:
            patch_results_base_dir = PATCH_APPLY_OUTPUT_DIR
        
        self.base_dir = Path(patch_results_base_dir)
        self._cache = {}
        self._valid_instances = {}
    
    def _parse_agent_name(self, agent_dir_name: str) -> tuple:
        """
        Parse agent directory name to extract framework and model
        
        Args:
            agent_dir_name: Directory name (e.g., "MagentLess_Claude-3.7-Sonnet")
            
        Returns:
            Tuple of (agent_framework, llm_model)
        """
        # Split by first underscore
        parts = agent_dir_name.split('_', 1)
        
        if len(parts) != 2:
            return "Unknown", "Unknown"
        
        agent_framework = parts[0]
        llm_model = parts[1]
        
        return agent_framework, llm_model
    
    def _load_agent_commit_mapping(self, agent_dir_name: str) -> List[Dict]:
        """
        Load agent_commit_mapping.json for an agent
        
        Args:
            agent_dir_name: Agent directory name
            
        Returns:
            List of commit mapping records
        """
        if agent_dir_name in self._cache:
            return self._cache[agent_dir_name]
        
        mapping_path = self.base_dir / agent_dir_name / "agent_commit_mapping.json"
        
        if not mapping_path.exists():
            self._cache[agent_dir_name] = []
            return []
        
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._cache[agent_dir_name] = data
            return data
        except Exception:
            self._cache[agent_dir_name] = []
            return []
    
    def get_valid_instances(self, agent_dir_name: str) -> List[str]:
        """
        Get list of instance IDs with status == "success"
        
        Args:
            agent_dir_name: Agent directory name
            
        Returns:
            List of valid instance IDs
        """
        if agent_dir_name in self._valid_instances:
            return self._valid_instances[agent_dir_name]
        
        mappings = self._load_agent_commit_mapping(agent_dir_name)
        
        valid_ids = [
            record['instance_id']
            for record in mappings
            if record.get('status') == 'success'
        ]
        
        self._valid_instances[agent_dir_name] = valid_ids
        return valid_ids
    
    def is_valid_instance(self, agent_dir_name: str, instance_id: str) -> bool:
        """
        Check if an instance has status == "success"
        
        Args:
            agent_dir_name: Agent directory name
            instance_id: Instance ID
            
        Returns:
            True if instance has status == "success", False otherwise
        """
        valid_ids = self.get_valid_instances(agent_dir_name)
        return instance_id in valid_ids
    
    def get_agent_info(self, agent_dir_name: str) -> Dict[str, str]:
        """
        Get agent framework and LLM model from agent directory name
        
        Args:
            agent_dir_name: Agent directory name (e.g., "MagentLess_Claude-3.7-Sonnet")
            
        Returns:
            Dictionary with 'agent_framework' and 'llm_model' keys
        """
        agent_framework, llm_model = self._parse_agent_name(agent_dir_name)
        
        return {
            'agent_framework': agent_framework,
            'llm_model': llm_model
        }
    
    def get_all_agent_dirs(self) -> List[str]:
        """
        Get all agent directory names
        
        Returns:
            List of agent directory names
        """
        if not self.base_dir.exists():
            return []
        
        return [
            d.name
            for d in self.base_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]


class IssueLengthLoader:
    """Loader for calculating issue text length in tokens"""
    
    def __init__(self, msb_java_dir: Optional[str] = None):
        """
        Initialize loader
        
        Args:
            msb_java_dir: Directory containing Multi-SWE-bench Java JSONL files
        """
        self.golden_loader = GoldenPatchLoader(msb_java_dir)
        self._tokenizer = None
        self._token_cache = {}
    
    def _get_tokenizer(self):
        """Lazy load tokenizer"""
        if self._tokenizer is None:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer
    
    def get_issue_length(self, instance_id: str) -> int:
        """
        Calculate token count of resolved issues
        
        Args:
            instance_id: Instance ID
            
        Returns:
            Total token count of all resolved issues' body text
        """
        if instance_id in self._token_cache:
            return self._token_cache[instance_id]
        
        # Get golden record
        record = self.golden_loader.get_golden_record(instance_id)
        
        if not record:
            self._token_cache[instance_id] = 0
            return 0
        
        # Get resolved_issues
        resolved_issues = record.get('resolved_issues', [])
        
        if not resolved_issues:
            self._token_cache[instance_id] = 0
            return 0
        
        # Calculate total token count
        tokenizer = self._get_tokenizer()
        total_tokens = 0
        
        for issue in resolved_issues:
            issue_body = issue.get('body', '')
            if issue_body:
                tokens = tokenizer.encode(issue_body)
                total_tokens += len(tokens)
        
        self._token_cache[instance_id] = total_tokens
        return total_tokens
