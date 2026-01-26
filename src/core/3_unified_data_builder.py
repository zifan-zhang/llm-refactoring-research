"""
Unified Data Builder

Integrates multiple research data sources to build a comprehensive analytical dataset
for statistical research on LLM agent performance and code refactoring.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import json


@dataclass
class RefactoringInfo:
    """Refactoring information data class"""
    
    type: str
    """Refactoring type (e.g., "Extract Method", "Rename Variable")"""
    
    description: str
    """Refactoring description"""
    
    left_locations: List[Dict[str, Any]]
    """Code locations before refactoring"""
    
    right_locations: List[Dict[str, Any]]
    """Code locations after refactoring"""


# Field group definitions for reducing repetition
IDENTIFIER_FIELDS = ['instance_id', 'agent_name', 'org', 'repo']
CATEGORICAL_VARS = ['task_difficulty', 'llm_model', 'agent_framework', 'issue_type']
NUMERICAL_VARS = [
    'issue_length', 'patch_size', 'file_coverage', 'line_coverage',
    'golden_patch_length', 'golden_patch_modified_files'
]
RESULT_VARS = [
    'num_turns', 'monetary_cost', 'input_token_consumption', 'output_token_consumption'
]
TARGET_VARS = ['is_compile_ok', 'is_issue_solved']


@dataclass
class UnifiedDataRow:
    """Unified data row - contains all fields required for analysis"""
    
    # ===== A. Identifier Fields =====
    instance_id: str
    agent_name: str
    org: str
    repo: str
    
    # ===== B. Refactoring Information Fields =====
    agent_has_refactoring: bool
    golden_has_refactoring: bool
    
    # ===== C. Independent Variable Fields (10 variables) =====
    task_difficulty: str
    llm_model: str
    agent_framework: str
    issue_type: str
    issue_length: int
    patch_size: int
    file_coverage: float
    line_coverage: float
    golden_patch_length: int
    golden_patch_modified_files: int
    
    # ===== D. Result Variable Fields (6 variables) =====
    num_turns: int
    is_compile_ok: bool
    is_issue_solved: str
    monetary_cost: float
    input_token_consumption: int
    output_token_consumption: int
    
    # ===== Fields with defaults (must be at the end) =====
    agent_refactoring_type_count: Dict[str, int] = field(default_factory=dict)
    agent_refactorings: List[RefactoringInfo] = field(default_factory=list)
    golden_refactoring_type_count: Dict[str, int] = field(default_factory=dict)
    golden_refactorings: List[RefactoringInfo] = field(default_factory=list)
    
    @staticmethod
    def _refactorings_to_dict(refactorings: List[RefactoringInfo]) -> List[Dict[str, Any]]:
        """Convert refactoring list to dictionary format"""
        return [
            {
                'type': r.type,
                'description': r.description,
                'left_locations': r.left_locations,
                'right_locations': r.right_locations
            }
            for r in refactorings
        ]
    
    @staticmethod
    def _dict_to_refactorings(data: List[Dict[str, Any]]) -> List[RefactoringInfo]:
        """Convert dictionary to refactoring list"""
        return [
            RefactoringInfo(
                type=r['type'],
                description=r['description'],
                left_locations=r['left_locations'],
                right_locations=r['right_locations']
            )
            for r in data
        ]
    
    def _get_refactoring_dict(self, prefix: str) -> Dict[str, Any]:
        """Get refactoring info dictionary for given prefix (agent/golden)"""
        has_ref = getattr(self, f'{prefix}_has_refactoring')
        type_count = getattr(self, f'{prefix}_refactoring_type_count')
        refactorings = getattr(self, f'{prefix}_refactorings')
        
        return {
            f'{prefix}_has_refactoring': has_ref,
            f'{prefix}_refactoring_type_count': type_count,
            f'{prefix}_refactorings': self._refactorings_to_dict(refactorings)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert data row to dictionary format"""
        result = {}
        
        # A. Identifier fields
        for field_name in IDENTIFIER_FIELDS:
            result[field_name] = getattr(self, field_name)
        
        # B. Refactoring info (agent + golden)
        result.update(self._get_refactoring_dict('agent'))
        result.update(self._get_refactoring_dict('golden'))
        
        # C. Independent variables
        for field_name in CATEGORICAL_VARS + NUMERICAL_VARS:
            result[field_name] = getattr(self, field_name)
        
        # D. Result variables
        for field_name in RESULT_VARS + TARGET_VARS:
            result[field_name] = getattr(self, field_name)
        
        return result
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Convert data row to flat dictionary format suitable for statistical analysis
        
        Returns:
            Dictionary with identifier fields, refactoring features (as JSON strings),
            categorical/numerical variables, and binary target variables (0/1 format)
        """
        result = {}
        
        # A. Identifier fields
        for field_name in IDENTIFIER_FIELDS:
            result[field_name] = getattr(self, field_name)
        
        # B. Refactoring features (JSON string format)
        for prefix in ['agent', 'golden']:
            has_ref = getattr(self, f'{prefix}_has_refactoring')
            type_count = getattr(self, f'{prefix}_refactoring_type_count')
            
            result[f'{prefix}_has_refactoring'] = 1 if has_ref else 0
            result[f'{prefix}_refactoring_type_count'] = json.dumps(
                type_count, ensure_ascii=False
            ) if type_count else '{}'
        
        # C. Independent variables
        for field_name in CATEGORICAL_VARS + NUMERICAL_VARS:
            result[field_name] = getattr(self, field_name)
        
        # D. Result variables (numerical)
        for field_name in RESULT_VARS:
            result[field_name] = getattr(self, field_name)
        
        # D. Target variables (binarized)
        result['is_compile_ok'] = 1 if self.is_compile_ok else 0
        result['is_issue_solved'] = 1 if self.is_issue_solved == 'resolved' else 0
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedDataRow':
        """Create data row instance from dictionary"""
        # Convert refactorings for both agent and golden
        agent_refactorings = cls._dict_to_refactorings(data.get('agent_refactorings', []))
        golden_refactorings = cls._dict_to_refactorings(data.get('golden_refactorings', []))
        
        # Build kwargs dict
        kwargs = {
            'agent_refactorings': agent_refactorings,
            'golden_refactorings': golden_refactorings,
            'agent_refactoring_type_count': data.get('agent_refactoring_type_count', {}),
            'golden_refactoring_type_count': data.get('golden_refactoring_type_count', {}),
        }
        
        # Add all other fields
        all_fields = (IDENTIFIER_FIELDS + ['agent_has_refactoring', 'golden_has_refactoring'] +
                     CATEGORICAL_VARS + NUMERICAL_VARS + RESULT_VARS + TARGET_VARS)
        
        for field_name in all_fields:
            kwargs[field_name] = data[field_name]
        
        return cls(**kwargs)


class UnifiedDataBuilder:
    """Unified data builder - integrates multiple data sources"""
    
    def __init__(self):
        """Initialize data builder and loaders"""
        self.data_rows: List[UnifiedDataRow] = []
        self._init_loaders()
    
    def _init_loaders(self):
        """Initialize all data loaders"""
        from src.data_loader import (
            TaskDifficultyLoader,
            IssueTypeLoader,
            RefactoringDataLoader,
            PatchDataLoader,
            GoldenPatchLoader,
            CompilationLogLoader,
            FinalReportLoader,
            PatchApplyResultsLoader,
            IssueLengthLoader
        )
        
        self.difficulty_loader = TaskDifficultyLoader()
        self.issue_type_loader = IssueTypeLoader()
        self.refactoring_loader = RefactoringDataLoader()
        self.patch_loader = PatchDataLoader()
        self.golden_patch_loader = GoldenPatchLoader()
        self.compilation_loader = CompilationLogLoader()
        self.final_report_loader = FinalReportLoader()
        self.patch_apply_loader = PatchApplyResultsLoader()
        self.issue_length_loader = IssueLengthLoader()
    
    # ===== Basic Operations =====
    
    def add_row(self, row: UnifiedDataRow) -> None:
        """Add a data row"""
        self.data_rows.append(row)
    
    def get_rows(self) -> List[UnifiedDataRow]:
        """Get all data rows"""
        return self.data_rows
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert all data rows to dictionary list"""
        return [row.to_dict() for row in self.data_rows]
    
    def clear(self) -> None:
        """Clear all data rows"""
        self.data_rows.clear()
    
    def __len__(self) -> int:
        """Return the number of data rows"""
        return len(self.data_rows)
    
    def __iter__(self):
        """Make the object iterable"""
        return iter(self.data_rows)
    
    # ===== Data Loading Methods =====
    
    def load_task_difficulty(self, instance_id: str) -> str:
        """Load task difficulty for instance"""
        return self.difficulty_loader.get_difficulty(instance_id)
    
    def load_issue_type(self, instance_id: str) -> str:
        """Load issue type for instance"""
        return self.issue_type_loader.get_issue_type(instance_id)
    
    def load_issue_length(self, instance_id: str) -> int:
        """Load issue length (token count) for instance"""
        return self.issue_length_loader.get_issue_length(instance_id)
    
    def _load_refactoring_info(self, ref_path) -> Tuple[bool, Dict[str, int], List[RefactoringInfo]]:
        """Helper method to load refactoring info from path"""
        from src.utils.refactoring_utils import process_refactoring_file
        return process_refactoring_file(str(ref_path))
    
    def load_agent_refactoring_info(self, agent_name: str, instance_id: str) -> Tuple[bool, Dict[str, int], List[RefactoringInfo]]:
        """Load agent refactoring information"""
        ref_path = self.refactoring_loader.get_agent_refactoring_path(agent_name, instance_id)
        return self._load_refactoring_info(ref_path)
    
    def load_golden_refactoring_info(self, instance_id: str) -> Tuple[bool, Dict[str, int], List[RefactoringInfo]]:
        """Load golden refactoring information"""
        ref_path = self.refactoring_loader.get_golden_refactoring_path(instance_id)
        return self._load_refactoring_info(ref_path)
    
    def load_patch_size(self, agent_name: str, instance_id: str) -> int:
        """Load patch size for agent"""
        from src.utils.patch_utils import get_patch_size
        
        patch_path = self.patch_loader.get_patch_path(agent_name, instance_id)
        return get_patch_size(str(patch_path))
    
    def load_golden_patch_info(self, instance_id: str) -> Tuple[int, int]:
        """Load golden patch information (patch_length, modified_files_count)"""
        from src.utils.patch_utils import parse_patch_content, get_modified_files_set
        
        golden_patch = self.golden_patch_loader.get_golden_patch(instance_id)
        
        if not golden_patch:
            return 0, 0
        
        stats = parse_patch_content(golden_patch)
        modified_files = get_modified_files_set(golden_patch)
        
        return stats['patch_size'], len(modified_files)
    
    def calculate_coverage(self, agent_name: str, instance_id: str) -> Tuple[float, float]:
        """Calculate file and line coverage between agent and golden patches"""
        from src.utils.patch_utils import calculate_file_coverage, calculate_line_coverage
        
        agent_patch = self.patch_loader.get_agent_patch_content(agent_name, instance_id)
        golden_patch = self.golden_patch_loader.get_golden_patch(instance_id)
        
        if not agent_patch or not golden_patch:
            return 0.0, 0.0
        
        file_cov = calculate_file_coverage(agent_patch, golden_patch)
        line_cov = calculate_line_coverage(agent_patch, golden_patch)
        
        return file_cov, line_cov
    
    def load_is_compile_ok(self, agent_name: str, instance_id: str) -> bool:
        """Load compilation status (True if succeeded)"""
        return self.compilation_loader.get_compilation_result(agent_name, instance_id)
    
    def load_is_issue_solved(self, agent_name: str, instance_id: str) -> str:
        """Load issue resolution status (resolved/unresolved)"""
        return self.final_report_loader.get_issue_status(agent_name, instance_id)
    
    def is_valid_instance(self, agent_name: str, instance_id: str) -> bool:
        """Check if instance has status == 'success' in patch apply results"""
        return self.patch_apply_loader.is_valid_instance(agent_name, instance_id)
    
    def get_valid_instances(self, agent_name: str) -> List[str]:
        """Get list of instance IDs with status == 'success' for an agent"""
        return self.patch_apply_loader.get_valid_instances(agent_name)
    
    def load_agent_framework(self, agent_name: str) -> str:
        """Load agent framework from agent name"""
        agent_info = self.patch_apply_loader.get_agent_info(agent_name)
        return agent_info['agent_framework']
    
    def load_llm_model(self, agent_name: str) -> str:
        """Load LLM model from agent name"""
        agent_info = self.patch_apply_loader.get_agent_info(agent_name)
        return agent_info['llm_model']
    
    def get_all_agents(self) -> List[str]:
        """Get all agent directory names"""
        return self.patch_apply_loader.get_all_agent_dirs()
    
    # ===== Export Methods =====
    
    def to_dataframe(self):
        """Convert all data rows to pandas DataFrame"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required. Install with: pip install pandas")
        
        data = [row.to_flat_dict() for row in self.data_rows]
        return pd.DataFrame(data)
    
    def export_to_csv(self, output_path: str) -> None:
        """Export data to CSV file"""
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)
        print(f"Successfully exported {len(df)} rows to {output_path}")
        print(f"Columns: {len(df.columns)}, Shape: {df.shape}")
    
    def get_refactoring_type_summary(self) -> Dict[str, Dict[str, int]]:
        """Get summary of all refactoring types found in the dataset"""
        result = {'agent': {}, 'golden': {}}
        
        for row in self.data_rows:
            for prefix in ['agent', 'golden']:
                type_count = getattr(row, f'{prefix}_refactoring_type_count')
                for ref_type, count in type_count.items():
                    result[prefix][ref_type] = result[prefix].get(ref_type, 0) + count
        
        return result


def _extract_org_repo(instance_id: str) -> Tuple[str, str]:
    """Extract org and repo from instance_id"""
    parts = instance_id.split('__')
    if len(parts) >= 2:
        org = parts[0]
        repo = parts[1].split('-')[0] if '-' in parts[1] else parts[1]
    else:
        org = "unknown"
        repo = "unknown"
    return org, repo


def build_complete_dataset(builder: UnifiedDataBuilder) -> None:
    """
    Build complete dataset by iterating through all agents and valid instances
    
    Args:
        builder: UnifiedDataBuilder instance
    """
    print("Building complete dataset...")
    print("="*80)
    
    all_agents = builder.get_all_agents()
    print(f"Found {len(all_agents)} agents: {', '.join(all_agents)}\n")
    
    total_rows = 0
    skipped_rows = 0
    
    for agent_name in all_agents:
        print(f"Processing agent: {agent_name}")
        
        valid_instances = builder.get_valid_instances(agent_name)
        print(f"  Valid instances: {len(valid_instances)}")
        
        agent_success = 0
        agent_skipped = 0
        
        for instance_id in valid_instances:
            try:
                org, repo = _extract_org_repo(instance_id)
                
                # Load refactoring info
                agent_has_ref, agent_ref_counts, agent_ref_list = builder.load_agent_refactoring_info(
                    agent_name, instance_id
                )
                golden_has_ref, golden_ref_counts, golden_ref_list = builder.load_golden_refactoring_info(
                    instance_id
                )
                
                # Load other fields
                task_difficulty = builder.load_task_difficulty(instance_id)
                issue_type = builder.load_issue_type(instance_id)
                issue_length = builder.load_issue_length(instance_id)
                patch_size = builder.load_patch_size(agent_name, instance_id)
                golden_patch_length, golden_modified_files = builder.load_golden_patch_info(instance_id)
                file_cov, line_cov = builder.calculate_coverage(agent_name, instance_id)
                is_compile_ok = builder.load_is_compile_ok(agent_name, instance_id)
                is_issue_solved = builder.load_is_issue_solved(agent_name, instance_id)
                llm_model = builder.load_llm_model(agent_name)
                agent_framework = builder.load_agent_framework(agent_name)
                
                # Create data row
                row = UnifiedDataRow(
                    # A. Identifiers
                    instance_id=instance_id,
                    agent_name=agent_name,
                    org=org,
                    repo=repo,
                    
                    # B. Refactoring info
                    agent_has_refactoring=agent_has_ref,
                    agent_refactoring_type_count=agent_ref_counts,
                    agent_refactorings=agent_ref_list,
                    golden_has_refactoring=golden_has_ref,
                    golden_refactoring_type_count=golden_ref_counts,
                    golden_refactorings=golden_ref_list,
                    
                    # C. Independent variables
                    task_difficulty=task_difficulty,
                    llm_model=llm_model,
                    agent_framework=agent_framework,
                    issue_type=issue_type,
                    issue_length=issue_length,
                    patch_size=patch_size,
                    file_coverage=file_cov,
                    line_coverage=line_cov,
                    golden_patch_length=golden_patch_length,
                    golden_patch_modified_files=golden_modified_files,
                    
                    # D. Result variables (TODO: load from agent trajectory for some fields)
                    num_turns=0,  # TODO: load from agent trajectory
                    is_compile_ok=is_compile_ok,
                    is_issue_solved=is_issue_solved,
                    monetary_cost=0.0,  # TODO: load from cost data
                    input_token_consumption=0,  # TODO: load from agent trajectory
                    output_token_consumption=0,  # TODO: load from agent trajectory
                )
                
                builder.add_row(row)
                agent_success += 1
                total_rows += 1
                
            except Exception as e:
                print(f"  ⚠ Skipped instance {instance_id}: {e}")
                agent_skipped += 1
                skipped_rows += 1
                continue
        
        print(f"  ✓ Success: {agent_success} rows, Skipped: {agent_skipped} rows")
        print(f"  Cumulative: {total_rows} rows\n")
    
    print("="*80)
    print(f"Dataset build complete!")
    print(f"  Total rows: {total_rows}")
    print(f"  Skipped rows: {skipped_rows}")
    print(f"  Success rate: {total_rows/(total_rows+skipped_rows)*100:.1f}%")


def main():
    """
    Main function: Build dataset and export to CSV
    
    Usage:
        python -m src.core.3_unified_data_builder
        or
        python src/core/3_unified_data_builder.py
    """
    from pathlib import Path
    
    print("\n" + "#"*80)
    print("# Unified Dataset Builder and Exporter")
    print("#"*80 + "\n")
    
    # Initialize builder
    builder = UnifiedDataBuilder()
    
    # Build complete dataset
    build_complete_dataset(builder)
    
    if len(builder) == 0:
        print("\n⚠ Warning: No data to export!")
        return 1
    
    # Create output directory - changed from "output" to "data"
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "data"
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("Exporting data...")
    print("="*80)
    
    # 1. Export CSV data
    csv_path = output_dir / "unified_data.csv"
    print(f"\nExporting to: {csv_path}")
    builder.export_to_csv(str(csv_path))
    print("✓ Data exported")
    
    # 2. Export refactoring type summary
    ref_summary = builder.get_refactoring_type_summary()
    summary_path = output_dir / "refactoring_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(ref_summary, f, indent=2, ensure_ascii=False)
    print(f"\nExporting to: {summary_path}")
    print("✓ Refactoring summary exported")
    
    # 3. Print statistics
    print("\n" + "="*80)
    print("Data Statistics Summary:")
    print("="*80)
    print(f"Total rows: {len(builder)}")
    
    df = builder.to_dataframe()
    print(f"Total columns: {len(df.columns)}")
    
    print("\nAgent Refactoring Type Distribution:")
    for ref_type, count in sorted(ref_summary['agent'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {ref_type:40s}: {count:>5d}")
    if len(ref_summary['agent']) > 10:
        print(f"  ... {len(ref_summary['agent'])-10} more types")
    
    print("\nGolden Refactoring Type Distribution:")
    for ref_type, count in sorted(ref_summary['golden'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {ref_type:40s}: {count:>5d}")
    if len(ref_summary['golden']) > 10:
        print(f"  ... {len(ref_summary['golden'])-10} more types")
    
    print("\nTarget Variable Distribution:")
    print(f"  Compilation success: {df['is_compile_ok'].sum()} / {len(df)} ({df['is_compile_ok'].mean()*100:.1f}%)")
    print(f"  Issue resolved: {df['is_issue_solved'].sum()} / {len(df)} ({df['is_issue_solved'].mean()*100:.1f}%)")
    
    print("\n" + "="*80)
    print("✓ Export complete!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {csv_path}")
    print(f"  - {summary_path}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
