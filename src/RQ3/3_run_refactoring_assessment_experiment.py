"""
Refactoring Assessment Experiment: Evaluating Necessity and Safety of Detected Refactorings

This experiment evaluates LLM-introduced refactorings in bug-fix patches along two dimensions:
- Necessity: Is the refactoring required for the fix to work?
- Safety: Is the refactoring correctly implemented?

Based on the assessment, it recommends actions: KEEP, REMOVE, or FIX.

Goal: Identify which refactorings should be preserved, removed, or fixed to optimize patches.
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

from src.constant import DATA_DIR, ROOT_DIR
from src.data_loader import (
    GoldenPatchLoader,
    PatchDataLoader,
    RefactoringDataLoader,
)
from src.RQ3.assessment_prompt import (
    AssessmentPromptPack,
    PromptMessages,
    CodeSnippet,
)

# Provider presets: maps provider name to default base_url and model
PROVIDER_CONFIGS: Dict[str, Dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.0-flash",
    },
}


@dataclass
class AssessmentInput:
    """Input data for a single refactoring assessment task"""
    instance_id: str
    agent_name: str
    issue_description: str
    code_snippets: List[CodeSnippet]  # Empty for now
    patch_diff: str
    refactoring_list: str


@dataclass
class RefactoringAssessment:
    """Assessment result for a single refactoring"""
    refactoring_type: str
    location: str
    necessity: str  # "required" or "unnecessary"
    safety: str  # "safe" or "unsafe"
    safety_issues: List[str]
    action: str  # "keep", "remove", or "fix"
    fix_suggestion: Optional[str]
    reasoning: str


@dataclass
class LLMAssessment:
    """Complete LLM assessment result for an instance"""
    instance_id: str
    agent_name: str
    overall_verdict: str  # "all_safe", "has_issues", or "uncertain"
    confidence: str  # "low", "medium", or "high"
    refactoring_assessments: List[RefactoringAssessment]
    summary: str
    actions_needed: Dict[str, int]  # {"keep": N, "remove": N, "fix": N}
    input_tokens: int
    output_tokens: int
    raw_response: str


@dataclass
class ExperimentStats:
    """Summary statistics for the experiment"""
    total_instances: int
    total_refactorings: int
    verdict_counts: Dict[str, int]  # "all_safe", "has_issues", "uncertain"
    action_counts: Dict[str, int]  # "keep", "remove", "fix"
    total_input_tokens: int
    total_output_tokens: int


class RefactoringAssessmentExperiment:
    """Main experiment class for refactoring assessment"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        provider: str = "deepseek",
        output_dir: Optional[Path] = None,
        include_code_context: bool = False,
    ):
        """
        Initialize experiment

        Args:
            api_key: API key for LLM
            model: Model name to use
            base_url: API base URL
            provider: Provider name (deepseek / openai / gemini), used for
                      provider-specific message formatting
            output_dir: Output directory for results
            include_code_context: Whether to include code context (not implemented yet)
        """
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.provider = provider.lower()
        self.include_code_context = include_code_context
        
        if output_dir is None:
            output_dir = ROOT_DIR / "output" / "RQ3" / "assessment_results"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data loaders
        self.golden_loader = GoldenPatchLoader()
        self.patch_loader = PatchDataLoader()
        self.refactoring_loader = RefactoringDataLoader()
        
        # Initialize prompt pack
        self.prompt_pack = AssessmentPromptPack()
        
        # Results storage
        self.results: List[LLMAssessment] = []
    
    def load_data(self, limit: Optional[int] = None) -> pd.DataFrame:
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
    
    def _prepare_assessment_input(
        self,
        instance_id: str,
        agent_name: str
    ) -> Optional[AssessmentInput]:
        """
        Prepare assessment input for a single instance
        
        Args:
            instance_id: Instance ID
            agent_name: Agent name
            
        Returns:
            AssessmentInput object or None if data is missing
        """
        # Get issue description
        issue_description = self.golden_loader.get_issue_description(instance_id)
        
        if not issue_description or issue_description == "[no issue description available]":
            print(f"  ⚠️  Missing issue description for {instance_id}")
            return None
        
        # Get patch diff
        patch_diff = self.patch_loader.get_agent_patch_content(agent_name, instance_id)
        
        if not patch_diff:
            print(f"  ⚠️  Missing patch file for {instance_id}")
            return None
        
        # Get refactoring list
        refactoring_list = self.refactoring_loader.format_refactoring_context(
            agent_name, instance_id
        )
        
        if not refactoring_list or refactoring_list == "[no refactoring context available]":
            print(f"  ⚠️  Missing refactoring data for {instance_id}")
            return None
        
        # Code context (not implemented yet, leave empty)
        code_snippets: List[CodeSnippet] = []
        
        return AssessmentInput(
            instance_id=instance_id,
            agent_name=agent_name,
            issue_description=issue_description,
            code_snippets=code_snippets,
            patch_diff=patch_diff,
            refactoring_list=refactoring_list,
        )
    
    def _call_llm(
        self,
        assessment_input: AssessmentInput,
        max_retries: int = 3
    ) -> Optional[LLMAssessment]:
        """
        Call LLM to assess refactorings with retry mechanism
        
        Args:
            assessment_input: Input data for assessment
            max_retries: Maximum number of retries
            
        Returns:
            LLMAssessment object or None if all retries failed
        """
        # Build prompt
        prompt_messages = self.prompt_pack.build_assessment_prompt(
            issue_description=assessment_input.issue_description,
            code_snippets=assessment_input.code_snippets or None,
            patch_diff=assessment_input.patch_diff,
            refactoring_list=assessment_input.refactoring_list,
        )

        # Gemini's OpenAI-compatible endpoint only supports a single system
        # message, so merge both system prompts into one.
        if self.provider == "gemini":
            merged_system = (
                prompt_messages.global_system
                + "\n\n"
                + prompt_messages.system
            )
            messages = [
                {"role": "system", "content": merged_system},
                {"role": "user", "content": prompt_messages.user},
            ]
            # Append the example as an extra user turn to avoid relying on
            # an assistant prefill that Gemini may not support.
            if prompt_messages.example_json:
                messages.append({
                    "role": "user",
                    "content": (
                        "For reference, here is an example of the expected "
                        "JSON output format:\n\n```json\n"
                        + prompt_messages.example_json
                        + "\n```"
                    ),
                })
        else:
            # Standard OpenAI-compatible message layout
            messages = [
                {"role": "system", "content": prompt_messages.global_system},
                {"role": "system", "content": prompt_messages.system},
                {"role": "user", "content": prompt_messages.user},
            ]
            if prompt_messages.example_json:
                messages.append({
                    "role": "assistant",
                    "content": (
                        "Here's an example of the expected JSON format:\n\n"
                        "```json\n" + prompt_messages.example_json + "\n```"
                    ),
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
                
                # Parse JSON response
                assessment_data = self._parse_llm_response(raw_response)
                
                if assessment_data is None:
                    print(f"  ⚠️  Failed to parse response (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        return None
                
                # Build RefactoringAssessment objects
                refactoring_assessments = []
                for ref_data in assessment_data.get('refactoring_assessments', []):
                    refactoring_assessments.append(RefactoringAssessment(
                        refactoring_type=ref_data.get('refactoring_type', ''),
                        location=ref_data.get('location', ''),
                        necessity=ref_data.get('necessity', ''),
                        safety=ref_data.get('safety', ''),
                        safety_issues=ref_data.get('safety_issues', []),
                        action=ref_data.get('action', ''),
                        fix_suggestion=ref_data.get('fix_suggestion'),
                        reasoning=ref_data.get('reasoning', ''),
                    ))
                
                # Build LLMAssessment
                return LLMAssessment(
                    instance_id=assessment_input.instance_id,
                    agent_name=assessment_input.agent_name,
                    overall_verdict=assessment_data.get('overall_verdict', 'uncertain'),
                    confidence=assessment_data.get('confidence', 'low'),
                    refactoring_assessments=refactoring_assessments,
                    summary=assessment_data.get('summary', ''),
                    actions_needed=assessment_data.get('actions_needed', {}),
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
                    return None
            
            except openai.APITimeoutError as e:
                print(f"  ⚠️  Timeout error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
            
            except openai.BadRequestError as e:
                # Context length error - cannot retry
                print(f"  ❌  Bad request (likely context too long): {e}")
                return None
            
            except Exception as e:
                print(f"  ❌  Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
        
        return None
    
    def _parse_llm_response(self, raw_response: str) -> Optional[Dict]:
        """
        Parse LLM response to extract JSON
        
        Args:
            raw_response: Raw response text from LLM
            
        Returns:
            Parsed dictionary or None if parsing failed
        """
        try:
            # Try to find JSON block in markdown code fence
            if "```json" in raw_response:
                start = raw_response.find("```json") + 7
                end = raw_response.find("```", start)
                json_str = raw_response[start:end].strip()
            elif "```" in raw_response:
                start = raw_response.find("```") + 3
                end = raw_response.find("```", start)
                json_str = raw_response[start:end].strip()
            else:
                # Try to parse the entire response as JSON
                json_str = raw_response.strip()
            
            # Parse JSON
            data = json.loads(json_str)
            return data
        
        except json.JSONDecodeError as e:
            print(f"  ❌  JSON parse error: {e}")
            return None
        except Exception as e:
            print(f"  ❌  Unexpected parse error: {e}")
            return None
    
    def run_experiment(self, limit: Optional[int] = None):
        """
        Run the full assessment experiment
        
        Args:
            limit: Maximum number of instances to process
        """
        print("\n" + "=" * 80)
        print("Starting Refactoring Assessment Experiment")
        print("=" * 80 + "\n")
        
        # Load data
        df = self.load_data(limit=limit)
        
        if len(df) == 0:
            print("No records to process!")
            return
        
        print(f"\nProcessing {len(df)} instances...\n")
        
        # Process each record
        for count, (_, row) in enumerate(df.iterrows(), 1):
            instance_id = row['instance_id']
            agent_name = row['agent_name']
            
            print(f"[{count}/{len(df)}] {instance_id} ({agent_name})")
            
            # Prepare input
            assessment_input = self._prepare_assessment_input(instance_id, agent_name)
            
            if assessment_input is None:
                print("  ⏭️  Skipping due to missing data\n")
                continue
            
            # Call LLM
            print("  🤖 Calling LLM for assessment...")
            result = self._call_llm(assessment_input)
            
            if result is None:
                print("  ❌ Assessment failed\n")
                continue
            
            # Store result
            self.results.append(result)
            
            print(f"  ✅ Assessment complete: {result.overall_verdict}")
            print(f"     Actions: keep={result.actions_needed.get('keep', 0)}, "
                  f"remove={result.actions_needed.get('remove', 0)}, "
                  f"fix={result.actions_needed.get('fix', 0)}")
            print(f"     Tokens: {result.input_tokens} in, {result.output_tokens} out\n")
        
        # Final summary
        print("\n" + "=" * 80)
        print("Experiment Complete")
        print("=" * 80 + "\n")
        
        stats = self._compute_stats()
        self._print_stats(stats)
        
        # Save final results
        self._save_results()
    
    def _compute_stats(self) -> ExperimentStats:
        """
        Compute summary statistics
        
        Returns:
            ExperimentStats object
        """
        total_instances = len(self.results)
        total_refactorings = sum(
            len(result.refactoring_assessments) for result in self.results
        )
        
        # Count verdicts
        verdict_counts = {}
        for result in self.results:
            verdict = result.overall_verdict
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        
        # Count actions
        action_counts = {"keep": 0, "remove": 0, "fix": 0}
        for result in self.results:
            for action, count in result.actions_needed.items():
                if action in action_counts:
                    action_counts[action] += count
        
        # Token usage
        total_input_tokens = sum(result.input_tokens for result in self.results)
        total_output_tokens = sum(result.output_tokens for result in self.results)
        
        return ExperimentStats(
            total_instances=total_instances,
            total_refactorings=total_refactorings,
            verdict_counts=verdict_counts,
            action_counts=action_counts,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
        )
    
    def _print_stats(self, stats: ExperimentStats):
        """Print summary statistics"""
        print(f"Total Instances Assessed: {stats.total_instances}")
        print(f"Total Refactorings Assessed: {stats.total_refactorings}")
        print()
        
        print("Overall Verdict Distribution:")
        for verdict, count in stats.verdict_counts.items():
            pct = 100 * count / stats.total_instances if stats.total_instances > 0 else 0
            print(f"  {verdict}: {count} ({pct:.1f}%)")
        print()
        
        print("Action Recommendations:")
        total_actions = sum(stats.action_counts.values())
        for action, count in stats.action_counts.items():
            pct = 100 * count / total_actions if total_actions > 0 else 0
            print(f"  {action.upper()}: {count} ({pct:.1f}%)")
        print()
        
        print("Token Usage:")
        print(f"  Input tokens: {stats.total_input_tokens:,}")
        print(f"  Output tokens: {stats.total_output_tokens:,}")
        print(f"  Total tokens: {stats.total_input_tokens + stats.total_output_tokens:,}")
        print()
    
    def _save_results(self):
        """Save results to JSON file"""
        if not self.results:
            print("No results to save!")
            return
        
        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.json"
        output_path = self.output_dir / filename
        
        # Compute stats
        stats = self._compute_stats()
        
        # Prepare output data
        output_data = {
            "metadata": {
                "model": self.model,
                "timestamp": timestamp,
                "include_code_context": self.include_code_context,
            },
            "summary_stats": asdict(stats),
            "results": [
                {
                    "instance_id": result.instance_id,
                    "agent_name": result.agent_name,
                    "overall_verdict": result.overall_verdict,
                    "confidence": result.confidence,
                    "summary": result.summary,
                    "actions_needed": result.actions_needed,
                    "refactoring_assessments": [
                        {
                            "refactoring_type": ref.refactoring_type,
                            "location": ref.location,
                            "necessity": ref.necessity,
                            "safety": ref.safety,
                            "safety_issues": ref.safety_issues,
                            "action": ref.action,
                            "fix_suggestion": ref.fix_suggestion,
                            "reasoning": ref.reasoning,
                        }
                        for ref in result.refactoring_assessments
                    ],
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


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run refactoring assessment experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Provider presets (--provider) and their defaults:",
            *[
                f"  {name:10s}  base-url={cfg['base_url']}  model={cfg['default_model']}"
                for name, cfg in PROVIDER_CONFIGS.items()
            ],
            "",
            "Examples:",
            "  # DeepSeek (default)",
            "  python -m src.RQ3.3_run_refactoring_assessment_experiment \\",
            "      --provider deepseek --api-key <KEY>",
            "",
            "  # Gemini",
            "  python -m src.RQ3.3_run_refactoring_assessment_experiment \\",
            "      --provider gemini --api-key <GEMINI_KEY>",
            "",
            "  # Gemini with a specific model",
            "  python -m src.RQ3.3_run_refactoring_assessment_experiment \\",
            "      --provider gemini --model gemini-1.5-pro --api-key <GEMINI_KEY>",
        ]),
    )

    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="API key for the chosen LLM provider",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="deepseek",
        choices=list(PROVIDER_CONFIGS.keys()),
        help=(
            "LLM provider preset; sets default base-url and model "
            "(default: deepseek)"
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model name to use. Defaults to the provider's default model "
            "when not specified."
        ),
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help=(
            "API base URL. Defaults to the provider's base URL when not "
            "specified."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of instances to process (for testing)",
    )

    parser.add_argument(
        "--include-code-context",
        action="store_true",
        help="Include code context (not implemented yet)",
    )

    args = parser.parse_args()

    # Resolve base_url and model from provider preset when not explicitly given
    provider_cfg = PROVIDER_CONFIGS[args.provider]
    resolved_base_url = args.base_url or provider_cfg["base_url"]
    resolved_model = args.model or provider_cfg["default_model"]

    print(f"Provider : {args.provider}")
    print(f"Model    : {resolved_model}")
    print(f"Base URL : {resolved_base_url}")

    # Create experiment
    experiment = RefactoringAssessmentExperiment(
        api_key=args.api_key,
        model=resolved_model,
        base_url=resolved_base_url,
        provider=args.provider,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        include_code_context=args.include_code_context,
    )

    # Run experiment
    experiment.run_experiment(limit=args.limit)


if __name__ == "__main__":
    main()
