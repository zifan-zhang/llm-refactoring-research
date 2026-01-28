"""
Patch Selection Experiment: Baseline vs Refactoring-aware selection

This experiment compares patch selection strategies:
- Baseline: select based on LLM scoring without refactoring context
- Refactoring-aware: select based on LLM scoring with refactoring context
- Random: randomly select a patch

Goal: Evaluate whether refactoring context helps select better patches that actually resolve issues.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import openai

from src.constant import DATA_DIR, ROOT_DIR
from src.data_loader import (
    GoldenPatchLoader,
    PatchDataLoader,
    RefactoringDataLoader,
    FinalReportLoader,
)
from src.RQ3.selection_prompt import PromptTemplatePack, PromptMessages, CodeSnippet


@dataclass
class CandidatePatch:
    """Single candidate patch in a selection group"""
    agent_name: str
    agent_framework: str
    patch_diff: str
    refactoring_context: str
    has_refactoring: bool
    # Ground truth labels
    is_resolved: bool
    is_compilable: bool
    file_coverage: float
    line_coverage: float


@dataclass
class SelectionInput:
    """Input data for a patch selection task"""
    instance_id: str
    llm_model: str
    issue_description: str
    code_snippets: List[CodeSnippet]
    candidates: List[CandidatePatch]


@dataclass
class ScoringResult:
    """Result of scoring a single candidate"""
    agent_name: str
    condition: str  # "baseline" or "refactoring_aware"
    resolution_score: int  # 0-100
    patch_resolution: str  # "resolves" or "does_not_resolve"
    confidence: str
    reasoning: str
    raw_response: str
    input_tokens: int
    output_tokens: int


@dataclass
class SelectionResult:
    """Result of a selection decision"""
    instance_id: str
    llm_model: str
    strategy: str  # "baseline", "refactoring_aware", or "random"
    selected_agent: str
    # Selected patch ground truth
    is_resolved: bool
    is_compilable: bool
    file_coverage: float
    line_coverage: float
    # Candidate info
    num_candidates: int
    num_frameworks: int
    # Scoring details (for non-random strategies)
    scoring_results: Optional[List[ScoringResult]] = None


@dataclass
class SelectionStats:
    """Statistics for a selection strategy"""
    strategy: str
    num_selections: int
    # Resolution metrics
    resolution_rate: float
    compilation_rate: float
    avg_file_coverage: float
    avg_line_coverage: float
    # Classification metrics (TP/FP/TN/FN with resolved as positive)
    tp: int  # Correctly selected resolved
    fp: int  # Selected unresolved thinking it's resolved
    tn: int  # Correctly identified as unresolved
    fn: int  # Missed resolved patches
    precision: float
    recall: float
    f1_score: float
    # Token usage (for LLM-based strategies)
    total_input_tokens: int
    total_output_tokens: int


class PatchSelectionExperiment:
    """Main experiment class for patch selection comparison"""
    
    @staticmethod
    def _create_empty_stats() -> Dict:
        """Create an empty statistics dictionary"""
        return {
            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
            'total_resolved': 0, 'total_compiled': 0, 
            'total_file_coverage': 0.0, 'total_line_coverage': 0.0,
            'total_input_tokens': 0, 'total_output_tokens': 0
        }
    
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        selection_size: int = 3,
        include_code_context: bool = False,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize experiment
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
            base_url: API base URL
            selection_size: Minimum number of candidates required
            include_code_context: Whether to include code context in prompts
            output_dir: Output directory for results
        """
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.selection_size = max(2, selection_size)
        self.include_code_context = include_code_context
        
        if output_dir is None:
            output_dir = ROOT_DIR / "output" / "RQ3" / "selection_experiment_results"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data loaders
        self.golden_loader = GoldenPatchLoader()
        self.patch_loader = PatchDataLoader()
        self.refactoring_loader = RefactoringDataLoader()
        self.report_loader = FinalReportLoader()
        
        # Initialize prompt template pack
        self.prompt_pack = PromptTemplatePack()
        
        # Results storage
        self.results: List[SelectionResult] = []
        
        # Statistics (use dict to reduce duplication)
        self.stats = {
            'baseline': self._create_empty_stats(),
            'refactoring_aware': self._create_empty_stats(),
            'random': self._create_empty_stats(),
        }
    
    def load_and_group_dataset(self, limit: Optional[int] = None) -> Dict[Tuple[str, str], List[Dict]]:
        """
        Load dataset and group by (instance_id, llm_model)
        
        Args:
            limit: Maximum number of groups to return (None for all)
            
        Returns:
            Dictionary mapping (instance_id, llm_model) to list of records
        """
        csv_path = DATA_DIR / "unified_data.csv"
        df = pd.read_csv(csv_path)
        
        print(f"Loaded {len(df)} total records from unified_data.csv")
        
        # Group by (instance_id, llm_model)
        grouped = defaultdict(list)
        for _, row in df.iterrows():
            key = (row['instance_id'], row['llm_model'])
            grouped[key].append(row.to_dict())
        
        print(f"Grouped into {len(grouped)} (instance_id, llm_model) combinations")
        
        # Apply filtering
        filtered_groups = {}
        for key, records in grouped.items():
            if self._should_include_group(key, records):
                filtered_groups[key] = records
        
        print(f"After filtering: {len(filtered_groups)} groups remain")
        
        if limit is not None:
            # Convert to list, take first N, convert back to dict
            items = list(filtered_groups.items())[:limit]
            filtered_groups = dict(items)
            print(f"Applied limit: {len(filtered_groups)} groups")
        
        return filtered_groups
    
    def _should_include_group(self, key: Tuple[str, str], records: List[Dict]) -> bool:
        """
        Check if a group should be included based on filtering criteria
        
        Args:
            key: (instance_id, llm_model) tuple
            records: List of records in this group
            
        Returns:
            True if group meets filtering criteria
        """
        # Must have at least selection_size candidates
        if len(records) < self.selection_size:
            return False
        
        # Must have at least 2 different frameworks
        frameworks = set(r['agent_framework'] for r in records)
        if len(frameworks) < 2:
            return False
        
        # Must not all be resolved (avoid trivial cases)
        all_resolved = all(r.get('is_issue_solved', 0) == 1 for r in records)
        if all_resolved:
            return False
        
        # Must have at least one candidate with refactoring
        has_any_refactoring = any(r.get('agent_has_refactoring', 0) == 1 for r in records)
        if not has_any_refactoring:
            return False
        
        return True
    
    def _prepare_selection_input(
        self,
        key: Tuple[str, str],
        records: List[Dict]
    ) -> Optional[SelectionInput]:
        """
        Prepare selection input from grouped records
        
        Args:
            key: (instance_id, llm_model) tuple
            records: List of records in this group
            
        Returns:
            SelectionInput or None if data is incomplete
        """
        instance_id, llm_model = key
        
        # Get issue description
        issue_description = self.golden_loader.get_issue_description(instance_id)
        if not issue_description or issue_description == "[no issue description available]":
            print(f"  Warning: No issue description for {instance_id}")
            return None
        
        # Get code snippets (留空，按需求先不实现)
        code_snippets = []
        
        # Build candidates
        candidates = []
        for record in records:
            agent_name = record['agent_name']
            
            # Get patch diff
            patch_diff = self.patch_loader.get_agent_patch_content(agent_name, instance_id)
            if not patch_diff:
                print(f"  Warning: No patch for {agent_name}/{instance_id}")
                continue
            
            # Get refactoring context
            refactoring_context = self.refactoring_loader.format_refactoring_context(
                agent_name, instance_id
            )
            has_refactoring = record.get('agent_has_refactoring', 0) == 1
            
            # Get ground truth labels
            is_resolved = record.get('is_issue_solved', 0) == 1
            is_compilable = record.get('is_compile_ok', 0) == 1
            file_coverage = float(record.get('file_coverage', 0.0))
            line_coverage = float(record.get('line_coverage', 0.0))
            
            candidates.append(CandidatePatch(
                agent_name=agent_name,
                agent_framework=record['agent_framework'],
                patch_diff=patch_diff,
                refactoring_context=refactoring_context,
                has_refactoring=has_refactoring,
                is_resolved=is_resolved,
                is_compilable=is_compilable,
                file_coverage=file_coverage,
                line_coverage=line_coverage,
            ))
        
        if len(candidates) < self.selection_size:
            print(f"  Warning: Only {len(candidates)} valid candidates (need {self.selection_size})")
            return None
        
        return SelectionInput(
            instance_id=instance_id,
            llm_model=llm_model,
            issue_description=issue_description,
            code_snippets=code_snippets,
            candidates=candidates,
        )
    
    def _build_selection_messages(
        self,
        selection_input: SelectionInput,
        candidate: CandidatePatch,
        include_refactoring: bool,
    ) -> List[Dict[str, str]]:
        """
        Build messages for LLM API call to score a candidate
        
        Args:
            selection_input: Input data for selection
            candidate: Candidate to score
            include_refactoring: Whether to include refactoring context
            
        Returns:
            List of message dictionaries for API
        """
        prompt_messages = self.prompt_pack.build_select_judgement_prompt(
            issue_description=selection_input.issue_description,
            code_snippets=selection_input.code_snippets if self.include_code_context else None,
            patch_diff=candidate.patch_diff,
            refactoring_context=candidate.refactoring_context,
            include_refactoring_context=include_refactoring,
        )
        
        messages = [
            {"role": "system", "content": prompt_messages.global_system},
            {"role": "system", "content": prompt_messages.system},
            {"role": "user", "content": prompt_messages.user},
        ]
        
        return messages
    
    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3,
    ) -> Tuple[str, int, int]:
        """
        Call LLM API with retry logic
        
        Args:
            messages: Messages for API
            max_retries: Maximum number of retries
            
        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=2048,
                )
                
                content = response.choices[0].message.content
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                
                return content, input_tokens, output_tokens
                
            except openai.APIError as e:
                if "context_length_exceeded" in str(e).lower():
                    print(f"    Context length exceeded, skipping")
                    raise
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"    API error (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"    Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"    Max retries reached")
                    raise
            
            except Exception as e:
                print(f"    Unexpected error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise
        
        raise RuntimeError("Failed to get LLM response after retries")
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, any]:
        """
        Parse LLM response to extract scoring fields
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            Dictionary with parsed fields including resolution_score
        """
        import re
        
        # Extract JSON from markdown code block
        json_match = re.search(r'```json\s*\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without markdown
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return {
                    'resolution_score': None,
                    'patch_resolution': 'does_not_resolve',
                    'confidence': 'low',
                    'reasoning': f'Failed to parse response: {response_text[:200]}...',
                }
        
        try:
            parsed = json.loads(json_str)
            
            # Validate resolution_score is present
            if 'resolution_score' not in parsed:
                return {
                    'resolution_score': None,
                    'patch_resolution': parsed.get('patch_resolution', 'does_not_resolve'),
                    'confidence': parsed.get('confidence', 'low'),
                    'reasoning': 'Missing resolution_score field',
                }
            
            return {
                'resolution_score': int(parsed['resolution_score']),
                'patch_resolution': parsed.get('patch_resolution', 'does_not_resolve'),
                'confidence': parsed.get('confidence', 'low'),
                'reasoning': parsed.get('reasoning', ''),
            }
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"    JSON parse error: {e}")
            return {
                'resolution_score': None,
                'patch_resolution': 'does_not_resolve',
                'confidence': 'low',
                'reasoning': f'Parse error: {str(e)}',
            }
    
    def _score_candidates(
        self,
        selection_input: SelectionInput,
        include_refactoring: bool,
    ) -> List[ScoringResult]:
        """
        Score all candidates using LLM
        
        Args:
            selection_input: Input data for selection
            include_refactoring: Whether to include refactoring context
            
        Returns:
            List of scoring results
        """
        condition = "refactoring_aware" if include_refactoring else "baseline"
        scoring_results = []
        
        for candidate in selection_input.candidates:
            try:
                messages = self._build_selection_messages(
                    selection_input, candidate, include_refactoring
                )
                response_text, input_tokens, output_tokens = self._call_llm(messages)
                parsed = self._parse_llm_response(response_text)
                
                # Skip if resolution_score is missing
                if parsed['resolution_score'] is None:
                    print(f"    Warning: Invalid response for {candidate.agent_name}, skipping")
                    continue
                
                scoring_results.append(ScoringResult(
                    agent_name=candidate.agent_name,
                    condition=condition,
                    resolution_score=parsed['resolution_score'],
                    patch_resolution=parsed['patch_resolution'],
                    confidence=parsed['confidence'],
                    reasoning=parsed['reasoning'],
                    raw_response=response_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ))
                
            except Exception as e:
                print(f"    Failed to score {candidate.agent_name}: {e}")
                continue
        
        return scoring_results
    
    def _select_by_score(
        self,
        candidates: List[CandidatePatch],
        scoring_results: List[ScoringResult],
    ) -> Optional[CandidatePatch]:
        """
        Select candidate with highest resolution_score
        
        Args:
            candidates: List of candidates
            scoring_results: List of scoring results
            
        Returns:
            Selected candidate or None if no valid scores
        """
        if not scoring_results:
            return None
        
        # Find highest score(s)
        max_score = max(sr.resolution_score for sr in scoring_results)
        top_scorers = [sr for sr in scoring_results if sr.resolution_score == max_score]
        
        # If tie, pick randomly
        selected_result = random.choice(top_scorers)
        
        # Find corresponding candidate
        for candidate in candidates:
            if candidate.agent_name == selected_result.agent_name:
                return candidate
        
        return None
    
    def _select_random(self, candidates: List[CandidatePatch]) -> CandidatePatch:
        """Randomly select a candidate"""
        return random.choice(candidates)
    
    def _run_selection_strategy(
        self,
        selection_input: SelectionInput,
        strategy: str,
        num_frameworks: int,
        include_refactoring: Optional[bool] = None,
    ) -> Optional[SelectionResult]:
        """
        Run a single selection strategy
        
        Args:
            selection_input: Input data for selection
            strategy: Strategy name ("baseline", "refactoring_aware", or "random")
            num_frameworks: Number of frameworks in candidates
            include_refactoring: Whether to include refactoring (None for random)
            
        Returns:
            SelectionResult or None if selection failed
        """
        instance_id = selection_input.instance_id
        llm_model = selection_input.llm_model
        
        print(f"  Running {strategy} selection...")
        
        try:
            if strategy == "random":
                selected = self._select_random(selection_input.candidates)
                scoring_results = None
                total_input = 0
                total_output = 0
            else:
                scoring_results = self._score_candidates(selection_input, include_refactoring)
                selected = self._select_by_score(selection_input.candidates, scoring_results)
                
                if not selected:
                    print(f"    Failed to select candidate")
                    return None
                
                total_input = sum(sr.input_tokens for sr in scoring_results)
                total_output = sum(sr.output_tokens for sr in scoring_results)
            
            result = SelectionResult(
                instance_id=instance_id,
                llm_model=llm_model,
                strategy=strategy,
                selected_agent=selected.agent_name,
                is_resolved=selected.is_resolved,
                is_compilable=selected.is_compilable,
                file_coverage=selected.file_coverage,
                line_coverage=selected.line_coverage,
                num_candidates=len(selection_input.candidates),
                num_frameworks=num_frameworks,
                scoring_results=scoring_results,
            )
            
            self._update_stats(self.stats[strategy], selected, total_input, total_output)
            print(f"    Selected: {selected.agent_name} (resolved: {selected.is_resolved})")
            
            return result
            
        except Exception as e:
            print(f"  {strategy.capitalize()} selection failed: {e}")
            return None
    
    def _update_stats(
        self,
        stats_dict: Dict,
        selected: CandidatePatch,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        """
        Update statistics for a selection
        
        Args:
            stats_dict: Statistics dictionary to update
            selected: Selected candidate
            input_tokens: Total input tokens used
            output_tokens: Total output tokens used
        """
        # Update resolution/compile/coverage metrics
        if selected.is_resolved:
            stats_dict['total_resolved'] += 1
            stats_dict['tp'] += 1
        else:
            stats_dict['fp'] += 1
        
        if selected.is_compilable:
            stats_dict['total_compiled'] += 1
        
        stats_dict['total_file_coverage'] += selected.file_coverage
        stats_dict['total_line_coverage'] += selected.line_coverage
        stats_dict['total_input_tokens'] += input_tokens
        stats_dict['total_output_tokens'] += output_tokens
    
    def _compute_metrics(self, stats_dict: Dict, num_samples: int, strategy: str) -> SelectionStats:
        """
        Compute evaluation metrics from statistics
        
        Args:
            stats_dict: Statistics dictionary
            num_samples: Total number of selections
            strategy: Strategy name
            
        Returns:
            SelectionStats object
        """
        tp = stats_dict['tp']
        fp = stats_dict['fp']
        tn = stats_dict['tn']
        fn = stats_dict['fn']
        
        resolution_rate = stats_dict['total_resolved'] / num_samples if num_samples > 0 else 0.0
        compilation_rate = stats_dict['total_compiled'] / num_samples if num_samples > 0 else 0.0
        avg_file_coverage = stats_dict['total_file_coverage'] / num_samples if num_samples > 0 else 0.0
        avg_line_coverage = stats_dict['total_line_coverage'] / num_samples if num_samples > 0 else 0.0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return SelectionStats(
            strategy=strategy,
            num_selections=num_samples,
            resolution_rate=resolution_rate,
            compilation_rate=compilation_rate,
            avg_file_coverage=avg_file_coverage,
            avg_line_coverage=avg_line_coverage,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            total_input_tokens=stats_dict['total_input_tokens'],
            total_output_tokens=stats_dict['total_output_tokens'],
        )
    
    def run(self, limit: Optional[int] = None):
        """
        Run the complete experiment
        
        Args:
            limit: Maximum number of groups to process
        """
        print("=" * 80)
        print("Patch Selection Experiment: Baseline vs Refactoring-aware vs Random")
        print("=" * 80)
        print(f"Model: {self.model}")
        print(f"Selection size: {self.selection_size}")
        print(f"Include code context: {self.include_code_context}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        # Load and group dataset
        grouped_data = self.load_and_group_dataset(limit=limit)
        print()
        
        # Process each group
        for i, (key, records) in enumerate(grouped_data.items(), 1):
            instance_id, llm_model = key
            print(f"[{i}/{len(grouped_data)}] Processing {instance_id} / {llm_model}")
            print(f"  Candidates: {len(records)}")
            
            # Prepare selection input
            selection_input = self._prepare_selection_input(key, records)
            if selection_input is None:
                print(f"  Skipping due to missing data")
                print()
                continue
            
            num_frameworks = len(set(c.agent_framework for c in selection_input.candidates))
            print(f"  Valid candidates: {len(selection_input.candidates)} from {num_frameworks} frameworks")
            
            # Run all three strategies
            for strategy, include_refactoring in [
                ("baseline", False),
                ("refactoring_aware", True),
                ("random", None),
            ]:
                result = self._run_selection_strategy(
                    selection_input, strategy, num_frameworks, include_refactoring
                )
                if result:
                    self.results.append(result)
            
            print()
        
        # Compute final metrics
        evaluations = {}
        for strategy in ["baseline", "refactoring_aware", "random"]:
            num_samples = len([r for r in self.results if r.strategy == strategy])
            evaluations[strategy] = self._compute_metrics(self.stats[strategy], num_samples, strategy)
        
        # Print summary
        self._print_summary(evaluations)
        
        # Save final results
        self.save_results(intermediate=False, evaluations=evaluations)
    
    def _print_summary(self, evaluations: Dict[str, SelectionStats]):
        """Print experiment summary"""
        print("=" * 80)
        print("EXPERIMENT RESULTS")
        print("=" * 80)
        print()
        
        # Print individual strategy results
        for strategy in ["baseline", "refactoring_aware", "random"]:
            eval_stats = evaluations[strategy]
            print(f"{eval_stats.strategy.upper().replace('_', ' ')}")
            print("-" * 40)
            print(f"  Selections: {eval_stats.num_selections}")
            print(f"  Resolution rate: {eval_stats.resolution_rate:.3f}")
            print(f"  Compilation rate: {eval_stats.compilation_rate:.3f}")
            print(f"  Avg file coverage: {eval_stats.avg_file_coverage:.3f}")
            print(f"  Avg line coverage: {eval_stats.avg_line_coverage:.3f}")
            print(f"  Classification metrics:")
            print(f"    Precision: {eval_stats.precision:.3f}")
            print(f"    Recall: {eval_stats.recall:.3f}")
            print(f"    F1 Score: {eval_stats.f1_score:.3f}")
            print(f"    TP: {eval_stats.tp}  FP: {eval_stats.fp}")
            print(f"    FN: {eval_stats.fn}  TN: {eval_stats.tn}")
            if eval_stats.total_input_tokens > 0:
                print(f"  Token usage:")
                print(f"    Input: {eval_stats.total_input_tokens:,}")
                print(f"    Output: {eval_stats.total_output_tokens:,}")
            print()
        
        # Print improvements
        self._print_improvement(
            "Refactoring-aware vs Baseline",
            evaluations["refactoring_aware"],
            evaluations["baseline"]
        )
        self._print_improvement(
            "Refactoring-aware vs Random",
            evaluations["refactoring_aware"],
            evaluations["random"]
        )
        
        # Print total token usage across all strategies
        total_input_tokens = sum(eval_stats.total_input_tokens for eval_stats in evaluations.values())
        total_output_tokens = sum(eval_stats.total_output_tokens for eval_stats in evaluations.values())
        total_tokens = total_input_tokens + total_output_tokens
        
        print("TOTAL TOKEN USAGE (All Strategies)")
        print("-" * 40)
        print(f"  Total Input Tokens: {total_input_tokens:,}")
        print(f"  Total Output Tokens: {total_output_tokens:,}")
        print(f"  Total Tokens: {total_tokens:,}")
        print()
    
    def _print_improvement(
        self,
        title: str,
        eval_a: SelectionStats,
        eval_b: SelectionStats,
    ):
        """Print improvement comparison between two evaluations"""
        print(f"IMPROVEMENT ({title})")
        print("-" * 40)
        print(f"  Resolution rate: {eval_a.resolution_rate - eval_b.resolution_rate:+.3f}")
        print(f"  Compilation rate: {eval_a.compilation_rate - eval_b.compilation_rate:+.3f}")
        print(f"  Avg file coverage: {eval_a.avg_file_coverage - eval_b.avg_file_coverage:+.3f}")
        print(f"  Avg line coverage: {eval_a.avg_line_coverage - eval_b.avg_line_coverage:+.3f}")
        print(f"  Precision: {eval_a.precision - eval_b.precision:+.3f}")
        print(f"  Recall: {eval_a.recall - eval_b.recall:+.3f}")
        print(f"  F1 Score: {eval_a.f1_score - eval_b.f1_score:+.3f}")
        print()
    
    def _compute_improvement_dict(
        self,
        eval_a: SelectionStats,
        eval_b: SelectionStats,
    ) -> Dict[str, float]:
        """Compute improvement metrics between two evaluations"""
        return {
            "resolution_rate": eval_a.resolution_rate - eval_b.resolution_rate,
            "compilation_rate": eval_a.compilation_rate - eval_b.compilation_rate,
            "avg_file_coverage": eval_a.avg_file_coverage - eval_b.avg_file_coverage,
            "avg_line_coverage": eval_a.avg_line_coverage - eval_b.avg_line_coverage,
            "precision": eval_a.precision - eval_b.precision,
            "recall": eval_a.recall - eval_b.recall,
            "f1_score": eval_a.f1_score - eval_b.f1_score,
        }
    
    def save_results(
        self,
        intermediate: bool = False,
        evaluations: Optional[Dict[str, SelectionStats]] = None,
    ):
        """
        Save experiment results to JSON file
        
        Args:
            intermediate: Whether this is an intermediate save
            evaluations: Dictionary of evaluation statistics by strategy
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if intermediate:
            filename = f"results_intermediate_{timestamp}.json"
        else:
            filename = f"results_final_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Convert results to dicts
        results_data = []
        for r in self.results:
            r_dict = asdict(r)
            # Convert scoring_results if present
            if r.scoring_results:
                r_dict['scoring_results'] = [asdict(sr) for sr in r.scoring_results]
            results_data.append(r_dict)
        
        # Build output structure
        output = {
            "experiment": "patch_selection_baseline_vs_refactoring_vs_random",
            "model": self.model,
            "selection_size": self.selection_size,
            "include_code_context": self.include_code_context,
            "timestamp": timestamp,
            "num_groups": len(self.results) // 3,  # Each group has 3 results
            "results": results_data,
        }
        
        if evaluations:
            output["evaluation"] = {
                strategy: asdict(eval_stats)
                for strategy, eval_stats in evaluations.items()
            }
            
            # Add improvement comparisons
            output["evaluation"]["improvement_vs_baseline"] = self._compute_improvement_dict(
                evaluations["refactoring_aware"], evaluations["baseline"]
            )
            output["evaluation"]["improvement_vs_random"] = self._compute_improvement_dict(
                evaluations["refactoring_aware"], evaluations["random"]
            )
            
            # Add total token usage
            total_input_tokens = sum(eval_stats.total_input_tokens for eval_stats in evaluations.values())
            total_output_tokens = sum(eval_stats.total_output_tokens for eval_stats in evaluations.values())
            output["evaluation"]["total_token_usage"] = {
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run patch selection experiment comparing baseline vs refactoring-aware vs random"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="OpenAI API key"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help="Model name (default: deepseek-chat)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://api.deepseek.com",
        help="API base URL (default: https://api.deepseek.com)"
    )
    parser.add_argument(
        "--selection-size",
        type=int,
        default=3,
        help="Minimum number of candidates required (default: 3, min: 2)"
    )
    parser.add_argument(
        "--include-code-context",
        action="store_true",
        help="Include code context in prompts (default: False)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of groups to process (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: output/RQ3/selection_experiment_results)"
    )
    
    args = parser.parse_args()
    
    # Create experiment
    experiment = PatchSelectionExperiment(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        selection_size=args.selection_size,
        include_code_context=args.include_code_context,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    
    # Run experiment
    experiment.run(limit=args.limit)


if __name__ == "__main__":
    main()
