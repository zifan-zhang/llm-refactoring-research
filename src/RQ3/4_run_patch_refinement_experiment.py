"""
Patch Refinement Experiment: Generating Refined Patches from Assessment Results

This experiment takes assessment results and generates refined patches by:
1. Keeping patches unchanged if all refactorings are marked as "keep"
2. Calling LLM to apply remove/fix operations to refine patches otherwise

Goal: Produce optimized patches with only necessary refactorings.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import openai
from tqdm import tqdm

from src.constant import DATA_DIR, ROOT_DIR
from src.data_loader import (
    GoldenPatchLoader,
    PatchDataLoader,
)
from src.RQ3.refinement_prompt import build_patch_refinement_prompt


@dataclass
class RefinementInput:
    """Input data for a single refinement task"""
    instance_id: str
    agent_name: str
    patch_diff: str
    refactoring_assessments: List[Dict]
    actions_summary: Dict[str, int]  # {"keep": N, "remove": N, "fix": N}


@dataclass
class RefinementResult:
    """Result of patch refinement for a single instance"""
    instance_id: str
    agent_name: str
    refinement_status: str  # "unchanged_keep", "llm_refined", "llm_context_limit", "llm_failed"
    actions_summary: Dict[str, int]
    original_patch: str
    refined_patch: str
    input_tokens: int
    output_tokens: int
    raw_response: Optional[str] = None


class PatchRefinementExperiment:
    """Main experiment class for patch refinement"""
    
    def __init__(
        self,
        api_key: str,
        assessment_results_path: Path,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        output_dir: Optional[Path] = None,
        api_delay: float = 0.5,
    ):
        """
        Initialize experiment
        
        Args:
            api_key: API key for LLM
            assessment_results_path: Path to assessment results JSON file
            model: Model name to use
            base_url: API base URL
            output_dir: Output directory for results
            api_delay: Delay between API calls in seconds
        """
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.api_delay = api_delay
        
        if output_dir is None:
            output_dir = ROOT_DIR / "output" / "RQ3" / "refinement_results"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load assessment results
        self.assessment_results_path = Path(assessment_results_path)
        self.assessment_results = self._load_assessment_results()
        
        # Initialize data loaders
        self.golden_loader = GoldenPatchLoader()
        self.patch_loader = PatchDataLoader()
        
        # Results storage
        self.results: List[RefinementResult] = []
    
    def _load_assessment_results(self) -> List[Dict]:
        """
        Load assessment results from JSON file
        
        Returns:
            List of assessment result dictionaries
        """
        print(f"Loading assessment results from: {self.assessment_results_path}")
        
        with open(self.assessment_results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        print(f"Loaded {len(results)} assessment results")
        
        return results
    
    def _load_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load and filter data from unified_data.csv
        
        Args:
            limit: Maximum number of records to process (for testing)
            
        Returns:
            Filtered DataFrame
        """
        print("Loading data from unified_data.csv...")
        
        csv_path = DATA_DIR / "unified_data.csv"
        df = pd.read_csv(csv_path)
        
        print(f"Total records loaded: {len(df)}")
        
        # Filter: only records with agent_has_refactoring=1
        df = df[df['agent_has_refactoring'] == 1].copy()
        print(f"Records with refactorings: {len(df)}")
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            df = df.head(limit)
            print(f"Limited to {len(df)} records for testing")
        
        return df
    
    def _prepare_refinement_input(
        self,
        assessment_result: Dict
    ) -> Optional[RefinementInput]:
        """
        Prepare refinement input from assessment result
        
        Args:
            assessment_result: Assessment result dictionary
            
        Returns:
            RefinementInput object or None if data is missing
        """
        instance_id = assessment_result.get('instance_id')
        agent_name = assessment_result.get('agent_name')
        
        if not instance_id or not agent_name:
            print(f"  ⚠️  Missing instance_id or agent_name")
            return None
        
        # Get original patch
        patch_diff = self.patch_loader.get_agent_patch_content(agent_name, instance_id)
        
        if not patch_diff:
            print(f"  ⚠️  Missing patch file for {instance_id}")
            return None
        
        # Get refactoring assessments and actions
        refactoring_assessments = assessment_result.get('refactoring_assessments', [])
        actions_summary = assessment_result.get('actions_needed', {})
        
        if not refactoring_assessments:
            print(f"  ⚠️  No refactoring assessments found for {instance_id}")
            return None
        
        return RefinementInput(
            instance_id=instance_id,
            agent_name=agent_name,
            patch_diff=patch_diff,
            refactoring_assessments=refactoring_assessments,
            actions_summary=actions_summary,
        )
    
    def _determine_refinement_status(self, actions_summary: Dict[str, int]) -> str:
        """
        Determine if refinement is needed based on actions
        
        Args:
            actions_summary: Dictionary with action counts
            
        Returns:
            "unchanged_keep" if no refinement needed, "needs_refinement" otherwise
        """
        remove_count = actions_summary.get('remove', 0)
        fix_count = actions_summary.get('fix', 0)
        
        if remove_count == 0 and fix_count == 0:
            return "unchanged_keep"
        else:
            return "needs_refinement"
    
    def _call_llm_for_refinement(
        self,
        refinement_input: RefinementInput,
        max_retries: int = 3
    ) -> Optional[RefinementResult]:
        """
        Call LLM to refine patch with retry mechanism
        
        Args:
            refinement_input: Input data for refinement
            max_retries: Maximum number of retries
            
        Returns:
            RefinementResult object or None if all retries failed
        """
        # Build prompt
        assessments_json = json.dumps(
            refinement_input.refactoring_assessments,
            indent=2,
            ensure_ascii=False
        )
        
        prompt_messages = build_patch_refinement_prompt(
            patch_diff=refinement_input.patch_diff,
            refactoring_assessments=assessments_json,
        )
        
        # Prepare messages for API
        messages = [
            {"role": "system", "content": prompt_messages.global_system},
            {"role": "system", "content": prompt_messages.system},
            {"role": "user", "content": prompt_messages.user},
        ]
        
        # Add example if available
        if prompt_messages.example_json:
            messages.append({
                "role": "assistant",
                "content": f"Here's an example of the expected diff format:\n\n{prompt_messages.example_json}"
            })
        
        # Retry loop with exponential backoff
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                )
                
                # Extract response content
                raw_response = response.choices[0].message.content
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                
                # Extract refined patch from response
                refined_patch = self._extract_diff_from_response(raw_response)
                
                if refined_patch is None:
                    print(f"  ⚠️  Failed to extract diff (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        return RefinementResult(
                            instance_id=refinement_input.instance_id,
                            agent_name=refinement_input.agent_name,
                            refinement_status="llm_failed",
                            actions_summary=refinement_input.actions_summary,
                            original_patch=refinement_input.patch_diff,
                            refined_patch="",
                            input_tokens=0,
                            output_tokens=0,
                            raw_response=raw_response,
                        )
                
                # Build successful result
                return RefinementResult(
                    instance_id=refinement_input.instance_id,
                    agent_name=refinement_input.agent_name,
                    refinement_status="llm_refined",
                    actions_summary=refinement_input.actions_summary,
                    original_patch=refinement_input.patch_diff,
                    refined_patch=refined_patch,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    raw_response=raw_response,
                )
                
            except openai.RateLimitError as e:
                print(f"  ⚠️  Rate limit error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    sleep_time = 2 ** (attempt + 1)
                    print(f"  ⏳  Waiting {sleep_time}s before retry...")
                    time.sleep(sleep_time)
                else:
                    return self._create_failed_result(
                        refinement_input, "llm_failed", "Rate limit exceeded"
                    )
            
            except openai.APITimeoutError as e:
                print(f"  ⚠️  Timeout error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return self._create_failed_result(
                        refinement_input, "llm_failed", "Timeout"
                    )
            
            except openai.BadRequestError as e:
                # Context length error - cannot retry
                print(f"  ❌  Bad request (likely context too long): {e}")
                return self._create_failed_result(
                    refinement_input, "llm_context_limit", str(e)
                )
            
            except Exception as e:
                print(f"  ❌  Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return self._create_failed_result(
                        refinement_input, "llm_failed", str(e)
                    )
        
        return self._create_failed_result(
            refinement_input, "llm_failed", "All retries exhausted"
        )
    
    def _create_failed_result(
        self,
        refinement_input: RefinementInput,
        status: str,
        error_msg: str
    ) -> RefinementResult:
        """Create a failed refinement result"""
        return RefinementResult(
            instance_id=refinement_input.instance_id,
            agent_name=refinement_input.agent_name,
            refinement_status=status,
            actions_summary=refinement_input.actions_summary,
            original_patch=refinement_input.patch_diff,
            refined_patch="",
            input_tokens=0,
            output_tokens=0,
            raw_response=f"Error: {error_msg}",
        )
    
    def _extract_diff_from_response(self, raw_response: str) -> Optional[str]:
        """
        Extract diff from LLM response
        
        Args:
            raw_response: Raw response text from LLM
            
        Returns:
            Extracted diff string or None if extraction failed
        """
        try:
            # Try to find diff block in markdown code fence
            if "```diff" in raw_response:
                start = raw_response.find("```diff") + 7
                end = raw_response.find("```", start)
                if end == -1:
                    print(f"  ⚠️  Found opening ```diff but no closing ```")
                    return None
                diff_str = raw_response[start:end].strip()
                if not diff_str:
                    print(f"  ⚠️  Extracted diff is empty")
                    return None
                return diff_str
            elif "```" in raw_response:
                start = raw_response.find("```") + 3
                end = raw_response.find("```", start)
                if end == -1:
                    print(f"  ⚠️  Found opening ``` but no closing ```")
                    return None
                diff_str = raw_response[start:end].strip()
                if not diff_str:
                    print(f"  ⚠️  Extracted diff is empty")
                    return None
                return diff_str
            else:
                # Try to use the entire response
                diff_str = raw_response.strip()
                if not diff_str:
                    print(f"  ⚠️  Response is empty")
                    return None
                return diff_str
        
        except Exception as e:
            print(f"  ❌  Diff extraction error: {e}")
            return None
    
    def run(self, limit: Optional[int] = None):
        """
        Run the full refinement experiment
        
        Args:
            limit: Maximum number of instances to process
        """
        print("\n" + "=" * 80)
        print("Starting Patch Refinement Experiment")
        print("=" * 80 + "\n")
        
        # Filter assessment results if limit is specified
        assessment_results = self.assessment_results
        if limit is not None and limit > 0:
            assessment_results = assessment_results[:limit]
            print(f"Limited to {len(assessment_results)} assessment results for testing")
        
        if len(assessment_results) == 0:
            print("No assessment results to process!")
            return
        
        print(f"\nProcessing {len(assessment_results)} assessment results...\n")
        
        # Process each assessment result with progress bar
        for assessment_result in tqdm(assessment_results, desc="Processing"):
            instance_id = assessment_result.get('instance_id', 'unknown')
            agent_name = assessment_result.get('agent_name', 'unknown')
            
            # Prepare input
            refinement_input = self._prepare_refinement_input(assessment_result)
            
            if refinement_input is None:
                print(f"\n⏭️  Skipping {instance_id} due to missing data")
                continue
            
            # Determine if refinement is needed
            status = self._determine_refinement_status(refinement_input.actions_summary)
            
            if status == "unchanged_keep":
                # No refinement needed - keep original patch
                result = RefinementResult(
                    instance_id=refinement_input.instance_id,
                    agent_name=refinement_input.agent_name,
                    refinement_status="unchanged_keep",
                    actions_summary=refinement_input.actions_summary,
                    original_patch=refinement_input.patch_diff,
                    refined_patch=refinement_input.patch_diff,  # Same as original
                    input_tokens=0,
                    output_tokens=0,
                    raw_response=None,
                )
                self.results.append(result)
                tqdm.write(f"✅ {instance_id}: unchanged (all keep)")
            else:
                # Need refinement - call LLM
                tqdm.write(f"🤖 {instance_id}: calling LLM for refinement...")
                result = self._call_llm_for_refinement(refinement_input)
                
                if result is None:
                    tqdm.write(f"❌ {instance_id}: refinement failed")
                    continue
                
                self.results.append(result)
                
                status_emoji = "✅" if result.refinement_status == "llm_refined" else "⚠️"
                tqdm.write(f"{status_emoji} {instance_id}: {result.refinement_status}")
                
                # API rate limiting delay
                if self.api_delay > 0:
                    time.sleep(self.api_delay)
        
        # Final summary
        print("\n" + "=" * 80)
        print("Experiment Complete")
        print("=" * 80 + "\n")
        
        self._print_summary()
        self._save_results()
    
    def _print_summary(self):
        """Print summary statistics"""
        total = len(self.results)
        
        if total == 0:
            print("No results to summarize!")
            return
        
        # Count statuses
        status_counts = {}
        for result in self.results:
            status = result.refinement_status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Token usage
        total_input_tokens = sum(r.input_tokens for r in self.results)
        total_output_tokens = sum(r.output_tokens for r in self.results)
        
        print(f"Total Instances Processed: {total}")
        print()
        
        print("Refinement Status Distribution:")
        for status, count in status_counts.items():
            pct = 100 * count / total if total > 0 else 0
            print(f"  {status}: {count} ({pct:.1f}%)")
        print()
        
        print("Token Usage:")
        print(f"  Input tokens: {total_input_tokens:,}")
        print(f"  Output tokens: {total_output_tokens:,}")
        print(f"  Total tokens: {total_input_tokens + total_output_tokens:,}")
        print()
    
    def _save_results(self):
        """Save results to JSON file"""
        if not self.results:
            print("No results to save!")
            return
        
        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"refinement_results_{timestamp}.json"
        output_path = self.output_dir / filename
        
        # Count statuses
        status_counts = {}
        for result in self.results:
            status = result.refinement_status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Token usage
        total_input_tokens = sum(r.input_tokens for r in self.results)
        total_output_tokens = sum(r.output_tokens for r in self.results)
        
        # Prepare output data
        output_data = {
            "metadata": {
                "model": self.model,
                "timestamp": timestamp,
                "assessment_results_file": str(self.assessment_results_path),
            },
            "summary": {
                "total": len(self.results),
                "status_counts": status_counts,
                "token_consumption": {
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "total_tokens": total_input_tokens + total_output_tokens,
                }
            },
            "results": [
                {
                    "instance_id": result.instance_id,
                    "agent_name": result.agent_name,
                    "refinement_status": result.refinement_status,
                    "actions_summary": result.actions_summary,
                    "original_patch": result.original_patch,
                    "refined_patch": result.refined_patch,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "raw_response": result.raw_response,
                }
                for result in self.results
            ],
        }
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_path}")


def find_latest_assessment_file(assessment_dir: Path) -> Optional[Path]:
    """
    Find the latest assessment results file in the directory
    
    Args:
        assessment_dir: Directory containing assessment results
        
    Returns:
        Path to latest file or None if not found
    """
    json_files = list(assessment_dir.glob("results_*.json"))
    
    if not json_files:
        return None
    
    # Sort by modification time
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    return latest_file


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run patch refinement experiment"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="API key for LLM service"
    )
    
    parser.add_argument(
        "--assessment-results",
        type=str,
        default=None,
        help="Path to assessment results JSON file (if not provided, uses latest)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help="Model to use (default: deepseek-chat)"
    )
    
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://api.deepseek.com",
        help="API base URL (default: https://api.deepseek.com)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of instances to process (for testing)"
    )
    
    parser.add_argument(
        "--api-delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # Determine assessment results file
    if args.assessment_results:
        assessment_results_path = Path(args.assessment_results)
    else:
        # Find latest assessment file
        assessment_dir = ROOT_DIR / "output" / "RQ3" / "assessment_results"
        assessment_results_path = find_latest_assessment_file(assessment_dir)
        
        if assessment_results_path is None:
            print(f"Error: No assessment results found in {assessment_dir}")
            print("Please specify --assessment-results or run assessment experiment first")
            return
        
        print(f"Using latest assessment results: {assessment_results_path}")
    
    if not assessment_results_path.exists():
        print(f"Error: Assessment results file not found: {assessment_results_path}")
        return
    
    # Create experiment
    experiment = PatchRefinementExperiment(
        api_key=args.api_key,
        assessment_results_path=assessment_results_path,
        model=args.model,
        base_url=args.base_url,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        api_delay=args.api_delay,
    )
    
    # Run experiment
    experiment.run(limit=args.limit)


if __name__ == "__main__":
    main()
