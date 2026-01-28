"""
Patch Judgement Experiment: Baseline vs Refactoring-aware LLM-as-a-judge

This experiment compares two LLM judging conditions:
- Baseline: judge sees only issue description and patch diff
- Refactoring-aware: judge additionally receives detected refactoring context

Goal: Evaluate whether refactoring context improves LLM's ability to judge 
      if a patch resolves the issue (with does_not_resolve as positive class).
"""

from __future__ import annotations

import argparse
import json
import time
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
from src.RQ3.prompt_template_pack import PromptTemplatePack, PromptMessages


@dataclass
class JudgementInput:
    """Input data for a single judgement task"""
    instance_id: str
    agent_name: str
    issue_description: str
    patch_diff: str
    refactoring_context: str
    true_label: str  # "resolved" or "unresolved"


@dataclass
class JudgementResult:
    """Result of a single LLM judgement"""
    instance_id: str
    agent_name: str
    condition: str  # "baseline" or "refactoring_aware"
    prediction: str  # "resolves" or "does_not_resolve"
    confidence: str  # "low", "medium", or "high"
    reasoning: str
    raw_response: str
    input_tokens: int
    output_tokens: int
    true_label: str
    is_correct: bool


@dataclass
class EvaluationStats:
    """Evaluation statistics for a condition"""
    condition: str
    num_samples: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    tp: int  # True Positives (correctly identified does_not_resolve)
    fp: int  # False Positives (incorrectly predicted does_not_resolve)
    tn: int  # True Negatives (correctly identified resolves)
    fn: int  # False Negatives (incorrectly predicted resolves when it does_not_resolve)
    total_input_tokens: int
    total_output_tokens: int


class PatchJudgementExperiment:
    """Main experiment class for patch judgement comparison"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize experiment
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
            base_url: API base URL
            output_dir: Output directory for results
        """
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        
        if output_dir is None:
            output_dir = ROOT_DIR / "output" / "RQ3" / "patch_judgement_experiment_results"
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
        self.results: List[JudgementResult] = []
        
        # Statistics
        self.baseline_stats = {
            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
            'total_input_tokens': 0, 'total_output_tokens': 0
        }
        self.refactoring_aware_stats = {
            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
            'total_input_tokens': 0, 'total_output_tokens': 0
        }
    
    def load_dataset_records(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Load dataset records from unified_data.csv
        
        Args:
            limit: Maximum number of records to load (None for all)
            
        Returns:
            List of filtered dataset records
        """
        csv_path = DATA_DIR / "unified_data.csv"
        df = pd.read_csv(csv_path)
        
        # Filter: only records with refactoring
        df_filtered = df[df['agent_has_refactoring'] == 1].copy()
        
        if limit is not None:
            df_filtered = df_filtered.head(limit)
        
        print(f"Loaded {len(df_filtered)} records with refactoring (out of {len(df)} total)")
        
        return df_filtered.to_dict('records')
    
    def _prepare_judgement_input(self, record: Dict) -> Optional[JudgementInput]:
        """
        Prepare judgement input from a dataset record
        
        Args:
            record: Dataset record dictionary
            
        Returns:
            JudgementInput or None if data is incomplete
        """
        instance_id = record['instance_id']
        agent_name = record['agent_name']
        
        # Get issue description
        issue_description = self.golden_loader.get_issue_description(instance_id)
        if not issue_description or issue_description == "[no issue description available]":
            print(f"Warning: No issue description for {instance_id}, skipping")
            return None
        
        # Get patch diff
        patch_diff = self.patch_loader.get_agent_patch_content(agent_name, instance_id)
        if not patch_diff:
            print(f"Warning: No patch content for {instance_id}/{agent_name}, skipping")
            return None
        
        # Get refactoring context
        refactoring_context = self.refactoring_loader.format_refactoring_context(
            agent_name, instance_id
        )
        
        # Get true label (whether issue was resolved)
        is_solved = record.get('is_issue_solved', 0)
        true_label = "resolved" if is_solved == 1 else "unresolved"
        
        return JudgementInput(
            instance_id=instance_id,
            agent_name=agent_name,
            issue_description=issue_description,
            patch_diff=patch_diff,
            refactoring_context=refactoring_context,
            true_label=true_label,
        )
    
    def _build_judgement_messages(
        self,
        judgement_input: JudgementInput,
        include_refactoring: bool,
    ) -> List[Dict[str, str]]:
        """
        Build messages for LLM API call
        
        Args:
            judgement_input: Input data for judgement
            include_refactoring: Whether to include refactoring context
            
        Returns:
            List of message dictionaries for API
        """
        prompt_messages = self.prompt_pack.build_patch_judgement_prompt(
            issue_description=judgement_input.issue_description,
            code_snippets=None,  # Not using code snippets in this experiment
            patch_diff=judgement_input.patch_diff,
            refactoring_context=judgement_input.refactoring_context,
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
                    print(f"Context length exceeded, skipping this instance")
                    raise
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Max retries reached, skipping")
                    raise
            
            except Exception as e:
                print(f"Unexpected error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise
        
        raise RuntimeError("Failed to get LLM response after retries")
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, str]:
        """
        Parse LLM response to extract structured fields
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            Dictionary with parsed fields
        """
        # Extract JSON from markdown code block
        import re
        
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
                    'patch_resolution': 'does_not_resolve',
                    'confidence': 'low',
                    'reasoning': f'Failed to parse response: {response_text[:200]}...',
                }
        
        try:
            parsed = json.loads(json_str)
            return {
                'patch_resolution': parsed.get('patch_resolution', 'does_not_resolve'),
                'confidence': parsed.get('confidence', 'low'),
                'reasoning': parsed.get('reasoning', ''),
            }
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return {
                'patch_resolution': 'does_not_resolve',
                'confidence': 'low',
                'reasoning': f'JSON parse error: {str(e)}',
            }
    
    def _update_stats(
        self,
        stats_dict: Dict,
        prediction: str,
        true_label: str,
        input_tokens: int,
        output_tokens: int,
    ):
        """
        Update statistics for a condition
        
        Args:
            stats_dict: Statistics dictionary to update
            prediction: Model prediction
            true_label: True label
            input_tokens: Input tokens used
            output_tokens: Output tokens used
        """
        # Convert labels to binary (positive class: does_not_resolve)
        pred_positive = prediction == "does_not_resolve"
        true_positive = true_label == "unresolved"
        
        if pred_positive and true_positive:
            stats_dict['tp'] += 1
        elif pred_positive and not true_positive:
            stats_dict['fp'] += 1
        elif not pred_positive and not true_positive:
            stats_dict['tn'] += 1
        else:  # not pred_positive and true_positive
            stats_dict['fn'] += 1
        
        stats_dict['total_input_tokens'] += input_tokens
        stats_dict['total_output_tokens'] += output_tokens
    
    def _compute_metrics(self, stats_dict: Dict, num_samples: int, condition: str) -> EvaluationStats:
        """
        Compute evaluation metrics from statistics
        
        Args:
            stats_dict: Statistics dictionary
            num_samples: Total number of samples
            condition: Condition name
            
        Returns:
            EvaluationStats object
        """
        tp = stats_dict['tp']
        fp = stats_dict['fp']
        tn = stats_dict['tn']
        fn = stats_dict['fn']
        
        accuracy = (tp + tn) / num_samples if num_samples > 0 else 0.0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return EvaluationStats(
            condition=condition,
            num_samples=num_samples,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            total_input_tokens=stats_dict['total_input_tokens'],
            total_output_tokens=stats_dict['total_output_tokens'],
        )
    
    def run(self, limit: Optional[int] = None):
        """
        Run the complete experiment
        
        Args:
            limit: Maximum number of records to process
        """
        print("=" * 80)
        print("Patch Judgement Experiment: Baseline vs Refactoring-aware")
        print("=" * 80)
        print(f"Model: {self.model}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        # Load dataset
        records = self.load_dataset_records(limit=limit)
        print()
        
        # Process each record
        for i, record in enumerate(records, 1):
            instance_id = record['instance_id']
            agent_name = record['agent_name']
            
            print(f"[{i}/{len(records)}] Processing {instance_id} / {agent_name}")
            
            # Prepare input
            judgement_input = self._prepare_judgement_input(record)
            if judgement_input is None:
                print(f"  Skipping due to missing data")
                print()
                continue
            
            # Run baseline judgement (without refactoring context)
            try:
                print(f"  Running baseline judgement...")
                baseline_messages = self._build_judgement_messages(
                    judgement_input, include_refactoring=False
                )
                baseline_response, baseline_input_tokens, baseline_output_tokens = self._call_llm(
                    baseline_messages
                )
                baseline_parsed = self._parse_llm_response(baseline_response)
                
                # Create baseline result
                baseline_result = JudgementResult(
                    instance_id=instance_id,
                    agent_name=agent_name,
                    condition="baseline",
                    prediction=baseline_parsed['patch_resolution'],
                    confidence=baseline_parsed['confidence'],
                    reasoning=baseline_parsed['reasoning'],
                    raw_response=baseline_response,
                    input_tokens=baseline_input_tokens,
                    output_tokens=baseline_output_tokens,
                    true_label=judgement_input.true_label,
                    is_correct=(
                        (baseline_parsed['patch_resolution'] == "resolves" and judgement_input.true_label == "resolved") or
                        (baseline_parsed['patch_resolution'] == "does_not_resolve" and judgement_input.true_label == "unresolved")
                    ),
                )
                self.results.append(baseline_result)
                self._update_stats(
                    self.baseline_stats,
                    baseline_parsed['patch_resolution'],
                    judgement_input.true_label,
                    baseline_input_tokens,
                    baseline_output_tokens,
                )
                print(f"    Baseline: {baseline_parsed['patch_resolution']} (confidence: {baseline_parsed['confidence']})")
                
            except Exception as e:
                print(f"  Baseline judgement failed: {e}")
            
            # Run refactoring-aware judgement
            try:
                print(f"  Running refactoring-aware judgement...")
                refactoring_messages = self._build_judgement_messages(
                    judgement_input, include_refactoring=True
                )
                refactoring_response, refactoring_input_tokens, refactoring_output_tokens = self._call_llm(
                    refactoring_messages
                )
                refactoring_parsed = self._parse_llm_response(refactoring_response)
                
                # Create refactoring-aware result
                refactoring_result = JudgementResult(
                    instance_id=instance_id,
                    agent_name=agent_name,
                    condition="refactoring_aware",
                    prediction=refactoring_parsed['patch_resolution'],
                    confidence=refactoring_parsed['confidence'],
                    reasoning=refactoring_parsed['reasoning'],
                    raw_response=refactoring_response,
                    input_tokens=refactoring_input_tokens,
                    output_tokens=refactoring_output_tokens,
                    true_label=judgement_input.true_label,
                    is_correct=(
                        (refactoring_parsed['patch_resolution'] == "resolves" and judgement_input.true_label == "resolved") or
                        (refactoring_parsed['patch_resolution'] == "does_not_resolve" and judgement_input.true_label == "unresolved")
                    ),
                )
                self.results.append(refactoring_result)
                self._update_stats(
                    self.refactoring_aware_stats,
                    refactoring_parsed['patch_resolution'],
                    judgement_input.true_label,
                    refactoring_input_tokens,
                    refactoring_output_tokens,
                )
                print(f"    Refactoring-aware: {refactoring_parsed['patch_resolution']} (confidence: {refactoring_parsed['confidence']})")
                
            except Exception as e:
                print(f"  Refactoring-aware judgement failed: {e}")
            
            print()
        
        # Compute final metrics
        num_baseline = len([r for r in self.results if r.condition == "baseline"])
        num_refactoring = len([r for r in self.results if r.condition == "refactoring_aware"])
        
        baseline_eval = self._compute_metrics(self.baseline_stats, num_baseline, "baseline")
        refactoring_eval = self._compute_metrics(
            self.refactoring_aware_stats, num_refactoring, "refactoring_aware"
        )
        
        # Print summary
        self._print_summary(baseline_eval, refactoring_eval)
        
        # Save final results
        self.save_results(
            intermediate=False,
            baseline_eval=baseline_eval,
            refactoring_eval=refactoring_eval,
        )
    
    def _print_summary(self, baseline_eval: EvaluationStats, refactoring_eval: EvaluationStats):
        """Print experiment summary"""
        print("=" * 80)
        print("EXPERIMENT RESULTS")
        print("=" * 80)
        print()
        
        print("BASELINE (without refactoring context)")
        print("-" * 40)
        print(f"  Samples: {baseline_eval.num_samples}")
        print(f"  Accuracy: {baseline_eval.accuracy:.3f}")
        print(f"  Precision: {baseline_eval.precision:.3f}")
        print(f"  Recall: {baseline_eval.recall:.3f}")
        print(f"  F1 Score: {baseline_eval.f1_score:.3f}")
        print(f"  Confusion Matrix:")
        print(f"    TP: {baseline_eval.tp}  FP: {baseline_eval.fp}")
        print(f"    FN: {baseline_eval.fn}  TN: {baseline_eval.tn}")
        print(f"  Token usage:")
        print(f"    Input: {baseline_eval.total_input_tokens:,}")
        print(f"    Output: {baseline_eval.total_output_tokens:,}")
        print()
        
        print("REFACTORING-AWARE (with refactoring context)")
        print("-" * 40)
        print(f"  Samples: {refactoring_eval.num_samples}")
        print(f"  Accuracy: {refactoring_eval.accuracy:.3f}")
        print(f"  Precision: {refactoring_eval.precision:.3f}")
        print(f"  Recall: {refactoring_eval.recall:.3f}")
        print(f"  F1 Score: {refactoring_eval.f1_score:.3f}")
        print(f"  Confusion Matrix:")
        print(f"    TP: {refactoring_eval.tp}  FP: {refactoring_eval.fp}")
        print(f"    FN: {refactoring_eval.fn}  TN: {refactoring_eval.tn}")
        print(f"  Token usage:")
        print(f"    Input: {refactoring_eval.total_input_tokens:,}")
        print(f"    Output: {refactoring_eval.total_output_tokens:,}")
        print()
        
        # Compute improvements
        acc_diff = refactoring_eval.accuracy - baseline_eval.accuracy
        prec_diff = refactoring_eval.precision - baseline_eval.precision
        recall_diff = refactoring_eval.recall - baseline_eval.recall
        f1_diff = refactoring_eval.f1_score - baseline_eval.f1_score
        
        print("IMPROVEMENT (Refactoring-aware vs Baseline)")
        print("-" * 40)
        print(f"  Accuracy: {acc_diff:+.3f}")
        print(f"  Precision: {prec_diff:+.3f}")
        print(f"  Recall: {recall_diff:+.3f}")
        print(f"  F1 Score: {f1_diff:+.3f}")
        print()
    
    def save_results(
        self,
        intermediate: bool = False,
        baseline_eval: Optional[EvaluationStats] = None,
        refactoring_eval: Optional[EvaluationStats] = None,
    ):
        """
        Save experiment results to JSON file
        
        Args:
            intermediate: Whether this is an intermediate save
            baseline_eval: Baseline evaluation statistics
            refactoring_eval: Refactoring-aware evaluation statistics
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if intermediate:
            filename = f"results_intermediate_{timestamp}.json"
        else:
            filename = f"results_final_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Convert results to dicts
        results_data = [asdict(r) for r in self.results]
        
        # Build output structure
        output = {
            "experiment": "patch_judgement_baseline_vs_refactoring",
            "model": self.model,
            "timestamp": timestamp,
            "num_instances": len(self.results) // 2,  # Each instance has 2 results
            "results": results_data,
        }
        
        if baseline_eval and refactoring_eval:
            output["evaluation"] = {
                "baseline": asdict(baseline_eval),
                "refactoring_aware": asdict(refactoring_eval),
                "improvement": {
                    "accuracy": refactoring_eval.accuracy - baseline_eval.accuracy,
                    "precision": refactoring_eval.precision - baseline_eval.precision,
                    "recall": refactoring_eval.recall - baseline_eval.recall,
                    "f1_score": refactoring_eval.f1_score - baseline_eval.f1_score,
                }
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run patch judgement experiment comparing baseline vs refactoring-aware conditions"
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
        "--limit",
        type=int,
        default=None,
        help="Limit number of records to process (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: output/RQ3/patch_judgement_experiment_results)"
    )
    
    args = parser.parse_args()
    
    # Create experiment
    experiment = PatchJudgementExperiment(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    
    # Run experiment
    experiment.run(limit=args.limit)


if __name__ == "__main__":
    main()
