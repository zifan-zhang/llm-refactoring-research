"""
Patch Selection Experiment: Using pre-computed judgement results

This experiment uses the judgement results from 1_run_patch_selection_experiment.py
to perform patch selection across 5 runs and compute average metrics.

Strategies:
- Baseline: select from patches without refactoring context
- Refactoring-aware: select from patches with refactoring context  
- Random: randomly select a patch

Selection logic:
1. Prefer patches judged as "resolves" (random if multiple)
2. If none, select from "does_not_resolve" patches (random if multiple)

Evaluation uses ground truth labels from unified_data.csv
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.constant import DATA_DIR, ROOT_DIR


@dataclass
class JudgementRecord:
    """Single judgement record from judgement experiment"""
    instance_id: str
    agent_name: str
    condition: str  # "baseline" or "refactoring_aware"
    prediction: str  # "resolves" or "does_not_resolve"
    confidence: str
    true_label: str


@dataclass
class GroundTruthRecord:
    """Ground truth record from unified_data.csv"""
    instance_id: str
    agent_name: str
    llm_model: str
    agent_framework: str
    is_resolved: bool
    is_compilable: bool
    file_coverage: float
    line_coverage: float


@dataclass
class SelectionResult:
    """Result of a single selection decision"""
    run_id: int
    instance_id: str
    llm_model: str
    strategy: str  # "baseline", "refactoring_aware", or "random"
    selected_agent: str
    selected_prediction: Optional[str]  # None for random
    # Ground truth metrics
    is_resolved: bool
    is_compilable: bool
    file_coverage: float
    line_coverage: float
    # Metadata
    num_candidates: int
    num_resolves: Optional[int]  # None for random
    num_does_not_resolve: Optional[int]  # None for random


@dataclass
class SelectionStats:
    """Aggregate statistics for a strategy"""
    strategy: str
    num_selections: int
    # Main metrics
    resolution_rate: float
    compilation_rate: float
    avg_file_coverage: float
    avg_line_coverage: float
    # Standard deviations across runs
    resolution_rate_std: float
    compilation_rate_std: float
    avg_file_coverage_std: float
    avg_line_coverage_std: float


class PatchSelectionExperiment:
    """Main experiment class for patch selection using judgement results"""
    
    def __init__(
        self,
        judgement_dir: Path,
        output_dir: Path,
        seed: int = 42,
    ):
        """
        Initialize experiment
        
        Args:
            judgement_dir: Directory containing judgement results
            output_dir: Output directory for selection results
            seed: Random seed for reproducibility
        """
        self.judgement_dir = judgement_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Load ground truth data
        self.ground_truth = self._load_ground_truth()
        print(f"Loaded {len(self.ground_truth)} ground truth records")
        
        # Results storage
        self.all_results: List[SelectionResult] = []
    
    def _load_ground_truth(self) -> Dict[Tuple[str, str], GroundTruthRecord]:
        """
        Load ground truth data from unified_data.csv
        
        Returns:
            Dictionary mapping (instance_id, agent_name) to ground truth record
        """
        csv_path = DATA_DIR / "unified_data.csv"
        df = pd.read_csv(csv_path)
        
        ground_truth = {}
        for _, row in df.iterrows():
            key = (row['instance_id'], row['agent_name'])
            ground_truth[key] = GroundTruthRecord(
                instance_id=row['instance_id'],
                agent_name=row['agent_name'],
                llm_model=row['llm_model'],
                agent_framework=row['agent_framework'],
                is_resolved=bool(row.get('is_issue_solved', 0) == 1),
                is_compilable=bool(row.get('is_compile_ok', 0) == 1),
                file_coverage=float(row.get('file_coverage', 0.0)),
                line_coverage=float(row.get('line_coverage', 0.0)),
            )
        
        return ground_truth
    
    def _load_judgement_results(self, result_file: Path) -> List[JudgementRecord]:
        """
        Load judgement results from a single result file
        
        Args:
            result_file: Path to judgement result JSON file
            
        Returns:
            List of judgement records
        """
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        records = []
        for result in data['results']:
            records.append(JudgementRecord(
                instance_id=result['instance_id'],
                agent_name=result['agent_name'],
                condition=result['condition'],
                prediction=result['prediction'],
                confidence=result['confidence'],
                true_label=result['true_label'],
            ))
        
        return records
    
    def _group_judgements_by_instance_llm(
        self,
        judgements: List[JudgementRecord]
    ) -> Dict[Tuple[str, str], List[JudgementRecord]]:
        """
        Group judgement records by (instance_id, llm_model)
        
        Args:
            judgements: List of judgement records
            
        Returns:
            Dictionary mapping (instance_id, llm_model) to list of judgements
        """
        grouped = defaultdict(list)
        
        for record in judgements:
            # Get llm_model from ground truth
            key = (record.instance_id, record.agent_name)
            if key not in self.ground_truth:
                continue
            
            llm_model = self.ground_truth[key].llm_model
            group_key = (record.instance_id, llm_model)
            grouped[group_key].append(record)
        
        return grouped
    
    def _filter_valid_groups(
        self,
        grouped: Dict[Tuple[str, str], List[JudgementRecord]]
    ) -> Dict[Tuple[str, str], List[JudgementRecord]]:
        """
        Filter groups to keep only those with:
        - At least 2 different agent frameworks
        - At least one baseline and one refactoring_aware judgement
        
        Args:
            grouped: Grouped judgement records
            
        Returns:
            Filtered dictionary
        """
        filtered = {}
        
        for key, records in grouped.items():
            instance_id, llm_model = key
            
            # Check framework diversity
            frameworks = set()
            for record in records:
                gt_key = (record.instance_id, record.agent_name)
                if gt_key in self.ground_truth:
                    frameworks.add(self.ground_truth[gt_key].agent_framework)
            
            if len(frameworks) < 2:
                continue
            
            # Check condition diversity
            conditions = set(r.condition for r in records)
            if 'baseline' not in conditions or 'refactoring_aware' not in conditions:
                continue
            
            filtered[key] = records
        
        return filtered
    
    def _select_by_judgement(
        self,
        judgements: List[JudgementRecord],
        condition: str
    ) -> Optional[JudgementRecord]:
        """
        Select a patch based on judgement results
        
        Logic:
        1. Filter judgements by condition (baseline or refactoring_aware)
        2. Prefer patches judged as "resolves"
        3. If none, select from "does_not_resolve"
        4. Random selection if multiple in same category
        
        Args:
            judgements: List of judgement records
            condition: "baseline" or "refactoring_aware"
            
        Returns:
            Selected judgement record or None if no valid candidates
        """
        # Filter by condition
        candidates = [j for j in judgements if j.condition == condition]
        if not candidates:
            return None
        
        # Separate by prediction
        resolves = [j for j in candidates if j.prediction == 'resolves']
        does_not_resolve = [j for j in candidates if j.prediction == 'does_not_resolve']
        
        # Prefer resolves, fallback to does_not_resolve
        if resolves:
            return random.choice(resolves)
        elif does_not_resolve:
            return random.choice(does_not_resolve)
        else:
            return None
    
    def _select_random(self, judgements: List[JudgementRecord]) -> Optional[JudgementRecord]:
        """
        Randomly select a patch from all candidates
        
        Args:
            judgements: List of judgement records
            
        Returns:
            Randomly selected judgement record
        """
        if not judgements:
            return None
        return random.choice(judgements)
    
    def _run_single_selection(
        self,
        run_id: int,
        grouped: Dict[Tuple[str, str], List[JudgementRecord]]
    ) -> List[SelectionResult]:
        """
        Run selection for all groups in a single judgement file
        
        Args:
            run_id: Run identifier (0-4)
            grouped: Grouped judgement records
            
        Returns:
            List of selection results
        """
        results = []
        
        for group_key, judgements in grouped.items():
            instance_id, llm_model = group_key
            
            # Count candidates by condition
            baseline_judgements = [j for j in judgements if j.condition == 'baseline']
            refactoring_judgements = [j for j in judgements if j.condition == 'refactoring_aware']
            
            # Run all three strategies
            for strategy in ['baseline', 'refactoring_aware', 'random']:
                if strategy == 'baseline':
                    selected = self._select_by_judgement(judgements, 'baseline')
                    if selected:
                        resolves = [j for j in baseline_judgements if j.prediction == 'resolves']
                        does_not = [j for j in baseline_judgements if j.prediction == 'does_not_resolve']
                        num_resolves = len(resolves)
                        num_does_not_resolve = len(does_not)
                    else:
                        num_resolves = None
                        num_does_not_resolve = None
                        
                elif strategy == 'refactoring_aware':
                    selected = self._select_by_judgement(judgements, 'refactoring_aware')
                    if selected:
                        resolves = [j for j in refactoring_judgements if j.prediction == 'resolves']
                        does_not = [j for j in refactoring_judgements if j.prediction == 'does_not_resolve']
                        num_resolves = len(resolves)
                        num_does_not_resolve = len(does_not)
                    else:
                        num_resolves = None
                        num_does_not_resolve = None
                        
                else:  # random
                    selected = self._select_random(judgements)
                    num_resolves = None
                    num_does_not_resolve = None
                
                if selected is None:
                    continue
                
                # Get ground truth
                gt_key = (selected.instance_id, selected.agent_name)
                if gt_key not in self.ground_truth:
                    continue
                
                gt = self.ground_truth[gt_key]
                
                # Create result
                result = SelectionResult(
                    run_id=run_id,
                    instance_id=instance_id,
                    llm_model=llm_model,
                    strategy=strategy,
                    selected_agent=selected.agent_name,
                    selected_prediction=selected.prediction if strategy != 'random' else None,
                    is_resolved=gt.is_resolved,
                    is_compilable=gt.is_compilable,
                    file_coverage=gt.file_coverage,
                    line_coverage=gt.line_coverage,
                    num_candidates=len(judgements),
                    num_resolves=num_resolves,
                    num_does_not_resolve=num_does_not_resolve,
                )
                
                results.append(result)
        
        return results
    
    def _compute_stats_per_run(
        self,
        results: List[SelectionResult],
        strategy: str
    ) -> Dict[str, float]:
        """
        Compute statistics for a single run and strategy
        
        Args:
            results: List of selection results
            strategy: Strategy name
            
        Returns:
            Dictionary of metrics
        """
        strategy_results = [r for r in results if r.strategy == strategy]
        
        if not strategy_results:
            return {
                'num_selections': 0,
                'resolution_rate': 0.0,
                'compilation_rate': 0.0,
                'avg_file_coverage': 0.0,
                'avg_line_coverage': 0.0,
            }
        
        num_selections = len(strategy_results)
        num_resolved = sum(1 for r in strategy_results if r.is_resolved)
        num_compiled = sum(1 for r in strategy_results if r.is_compilable)
        total_file_coverage = sum(r.file_coverage for r in strategy_results)
        total_line_coverage = sum(r.line_coverage for r in strategy_results)
        
        return {
            'num_selections': num_selections,
            'resolution_rate': num_resolved / num_selections,
            'compilation_rate': num_compiled / num_selections,
            'avg_file_coverage': total_file_coverage / num_selections,
            'avg_line_coverage': total_line_coverage / num_selections,
        }
    
    def _compute_aggregate_stats(
        self,
        all_results: List[SelectionResult],
        strategy: str
    ) -> SelectionStats:
        """
        Compute aggregate statistics across all runs
        
        Args:
            all_results: All selection results from all runs
            strategy: Strategy name
            
        Returns:
            SelectionStats with mean and std
        """
        # Get unique run IDs
        run_ids = sorted(set(r.run_id for r in all_results))
        
        # Compute per-run stats
        per_run_stats = []
        for run_id in run_ids:
            run_results = [r for r in all_results if r.run_id == run_id]
            stats = self._compute_stats_per_run(run_results, strategy)
            per_run_stats.append(stats)
        
        # Compute mean and std
        num_selections = int(np.mean([s['num_selections'] for s in per_run_stats]))
        
        resolution_rates = [s['resolution_rate'] for s in per_run_stats]
        compilation_rates = [s['compilation_rate'] for s in per_run_stats]
        file_coverages = [s['avg_file_coverage'] for s in per_run_stats]
        line_coverages = [s['avg_line_coverage'] for s in per_run_stats]
        
        return SelectionStats(
            strategy=strategy,
            num_selections=num_selections,
            resolution_rate=float(np.mean(resolution_rates)),
            compilation_rate=float(np.mean(compilation_rates)),
            avg_file_coverage=float(np.mean(file_coverages)),
            avg_line_coverage=float(np.mean(line_coverages)),
            resolution_rate_std=float(np.std(resolution_rates)),
            compilation_rate_std=float(np.std(compilation_rates)),
            avg_file_coverage_std=float(np.std(file_coverages)),
            avg_line_coverage_std=float(np.std(line_coverages)),
        )
    
    def run(self):
        """Run the complete experiment across all judgement files"""
        print("=" * 80)
        print("Patch Selection Experiment: Using Judgement Results")
        print("=" * 80)
        print(f"Judgement directory: {self.judgement_dir}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        # Find all judgement result files
        result_files = sorted(self.judgement_dir.glob("results_final_*.json"))
        print(f"Found {len(result_files)} judgement result files:")
        for f in result_files:
            print(f"  - {f.name}")
        print()
        
        if len(result_files) == 0:
            print("Error: No judgement result files found!")
            return
        
        # Process each judgement file as a separate run
        for run_id, result_file in enumerate(result_files):
            print(f"Processing run {run_id + 1}/{len(result_files)}: {result_file.name}")
            
            # Load judgements
            judgements = self._load_judgement_results(result_file)
            print(f"  Loaded {len(judgements)} judgement records")
            
            # Group by (instance_id, llm_model)
            grouped = self._group_judgements_by_instance_llm(judgements)
            print(f"  Grouped into {len(grouped)} (instance_id, llm_model) combinations")
            
            # Filter valid groups
            filtered = self._filter_valid_groups(grouped)
            print(f"  After filtering: {len(filtered)} valid groups")
            
            # Run selection
            run_results = self._run_single_selection(run_id, filtered)
            print(f"  Generated {len(run_results)} selection results")
            print()
            
            self.all_results.extend(run_results)
        
        print(f"Total selection results across all runs: {len(self.all_results)}")
        print()
        
        # Compute aggregate statistics
        print("Computing aggregate statistics...")
        stats = {}
        for strategy in ['baseline', 'refactoring_aware', 'random']:
            stats[strategy] = self._compute_aggregate_stats(self.all_results, strategy)
        
        # Print summary
        self._print_summary(stats)
        
        # Save results
        self.save_results(stats)
    
    def _print_summary(self, stats: Dict[str, SelectionStats]):
        """Print experiment summary"""
        print("=" * 80)
        print("EXPERIMENT RESULTS (Averaged across runs)")
        print("=" * 80)
        print()
        
        for strategy in ['baseline', 'refactoring_aware', 'random']:
            s = stats[strategy]
            print(f"{s.strategy.upper().replace('_', ' ')}")
            print("-" * 40)
            print(f"  Num selections: {s.num_selections}")
            print(f"  Resolution rate: {s.resolution_rate:.4f} ± {s.resolution_rate_std:.4f}")
            print(f"  Compilation rate: {s.compilation_rate:.4f} ± {s.compilation_rate_std:.4f}")
            print(f"  Avg file coverage: {s.avg_file_coverage:.4f} ± {s.avg_file_coverage_std:.4f}")
            print(f"  Avg line coverage: {s.avg_line_coverage:.4f} ± {s.avg_line_coverage_std:.4f}")
            print()
        
        # Print improvements
        self._print_improvement(
            "Refactoring-aware vs Baseline",
            stats['refactoring_aware'],
            stats['baseline']
        )
        self._print_improvement(
            "Refactoring-aware vs Random",
            stats['refactoring_aware'],
            stats['random']
        )
        self._print_improvement(
            "Baseline vs Random",
            stats['baseline'],
            stats['random']
        )
    
    def _print_improvement(
        self,
        title: str,
        stats_a: SelectionStats,
        stats_b: SelectionStats
    ):
        """Print improvement comparison"""
        print(f"IMPROVEMENT ({title})")
        print("-" * 40)
        print(f"  Resolution rate: {stats_a.resolution_rate - stats_b.resolution_rate:+.4f}")
        print(f"  Compilation rate: {stats_a.compilation_rate - stats_b.compilation_rate:+.4f}")
        print(f"  Avg file coverage: {stats_a.avg_file_coverage - stats_b.avg_file_coverage:+.4f}")
        print(f"  Avg line coverage: {stats_a.avg_line_coverage - stats_b.avg_line_coverage:+.4f}")
        print()
    
    def save_results(self, stats: Dict[str, SelectionStats]):
        """Save experiment results to JSON"""
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"selection_results_{timestamp}.json"
        
        # Convert results to dicts
        results_data = [asdict(r) for r in self.all_results]
        
        # Build output structure
        output = {
            "experiment": "patch_selection_using_judgement_results",
            "timestamp": timestamp,
            "num_runs": len(set(r.run_id for r in self.all_results)),
            "total_selections": len(self.all_results),
            "aggregate_statistics": {
                strategy: asdict(stat)
                for strategy, stat in stats.items()
            },
            "improvements": {
                "refactoring_aware_vs_baseline": {
                    "resolution_rate": stats['refactoring_aware'].resolution_rate - stats['baseline'].resolution_rate,
                    "compilation_rate": stats['refactoring_aware'].compilation_rate - stats['baseline'].compilation_rate,
                    "avg_file_coverage": stats['refactoring_aware'].avg_file_coverage - stats['baseline'].avg_file_coverage,
                    "avg_line_coverage": stats['refactoring_aware'].avg_line_coverage - stats['baseline'].avg_line_coverage,
                },
                "refactoring_aware_vs_random": {
                    "resolution_rate": stats['refactoring_aware'].resolution_rate - stats['random'].resolution_rate,
                    "compilation_rate": stats['refactoring_aware'].compilation_rate - stats['random'].compilation_rate,
                    "avg_file_coverage": stats['refactoring_aware'].avg_file_coverage - stats['random'].avg_file_coverage,
                    "avg_line_coverage": stats['refactoring_aware'].avg_line_coverage - stats['random'].avg_line_coverage,
                },
                "baseline_vs_random": {
                    "resolution_rate": stats['baseline'].resolution_rate - stats['random'].resolution_rate,
                    "compilation_rate": stats['baseline'].compilation_rate - stats['random'].compilation_rate,
                    "avg_file_coverage": stats['baseline'].avg_file_coverage - stats['random'].avg_file_coverage,
                    "avg_line_coverage": stats['baseline'].avg_line_coverage - stats['random'].avg_line_coverage,
                },
            },
            "detailed_results": results_data,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_path}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run patch selection experiment using pre-computed judgement results"
    )
    parser.add_argument(
        "--judgement-dir",
        type=str,
        default=None,
        help="Directory containing judgement results (default: output/RQ3/patch_judgement_experiment_results)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for selection results (default: output/RQ3/selection_experiment_results)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set default paths
    judgement_dir = Path(args.judgement_dir) if args.judgement_dir else \
        ROOT_DIR / "output" / "RQ3" / "patch_judgement_experiment_results"
    output_dir = Path(args.output_dir) if args.output_dir else \
        ROOT_DIR / "output" / "RQ3" / "selection_experiment_results"
    
    # Create and run experiment
    experiment = PatchSelectionExperiment(
        judgement_dir=judgement_dir,
        output_dir=output_dir,
        seed=args.seed,
    )
    
    experiment.run()


if __name__ == "__main__":
    main()
