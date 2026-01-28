"""Extracted select-judgement prompts for RQ3 selection experiment.

This file is a standalone copy of the selection-related prompt templates and
builders from `RQ3/prompt_template_pack.py`. It is used to isolate the prompts
and construction logic for the judge-and-select experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


GLOBAL_SYSTEM_PROMPT = """
You are a senior Java software engineering expert specializing in bug-fix review and refactoring optimization. Your goal is to ensure patches are correct, minimal, and contain only necessary changes.

Key principles:

1. Correctness first: the fix must resolve the reported issue
2. Minimize change scope: avoid unnecessary modifications
3. Refactoring discipline: keep only refactorings that directly support the fix or are tightly coupled to it
4. Maintain backward compatibility: do not introduce breaking changes to public APIs unless absolutely required
5. Preserve test interfaces: avoid changes that break existing test contracts

Note: Output format requirements are specified in the phase-specific system prompts.
""".strip()

SELECT_JUDGEMENT_SYSTEM_PROMPT_WITHOUT_REFACTORING = """
You are an impartial senior Java engineer acting as a correctness judge. Your goal is to assess how likely a candidate patch resolves the reported issue.

Evaluation principles:
1. Focus on the functional intent described in the issue.
2. Use the provided code context and diff to assess correctness.
3. Be SKEPTICAL by default. Start from a neutral baseline (50) and require concrete evidence to move the score up or down.
4. Provide a numerical score (0-100) representing the likelihood that the patch resolves the issue:
   - 80-100: Strong, unambiguous evidence of complete fix with no concerns
   - 65-79: Clear fix logic present, but minor gaps or uncertainties remain
   - 50-64: Plausible fix attempt, but unclear if it fully addresses the root cause
   - 35-49: Weak or incomplete fix, significant doubts about correctness
   - 0-34: Does not address the issue, introduces breakage, or is clearly wrong

Output requirements:
- Respond ONLY with a fenced JSON block (```json ... ```).
- Include the fields:
  * "resolution_score": integer 0-100 indicating likelihood of resolving the issue
  * "patch_resolution": "resolves" (score >= 50) or "does_not_resolve" (score < 50)
  * "confidence": "low", "medium", or "high"
  * "reasoning": structured explanation listing fix evidence, concerns, and score justification
""".strip()


SELECT_JUDGEMENT_SYSTEM_PROMPT_WITH_REFACTORING = """
You are an impartial senior Java engineer acting as a correctness judge. Your goal is to assess how likely a candidate patch resolves the reported issue, using refactoring context to identify potential risks.

Evaluation principles:
1. Focus on the functional intent described in the issue—the primary question is whether the core bug is fixed.
2. Use the provided code context and diff to assess correctness.
3. Be SKEPTICAL by default. Start from a neutral baseline (50) and require concrete evidence to move the score up or down.
4. Provide a numerical BASE score (0-100) representing the likelihood that the patch resolves the issue:
   - 80-100: Strong, unambiguous evidence of complete fix with no concerns
   - 65-79: Clear fix logic present, but minor gaps or uncertainties remain
   - 50-64: Plausible fix attempt, but unclear if it fully addresses the root cause
   - 35-49: Weak or incomplete fix, significant doubts about correctness
   - 0-34: Does not address the issue, introduces breakage, or is clearly wrong

5. CRITICAL: Refactoring in a bug-fix patch is a RED FLAG by default. Refactorings detected in the patch indicate changes beyond the minimal fix, which increases risk. Apply NEGATIVE adjustments unless you can prove the refactoring is absolutely necessary:

   DEFAULT (most refactorings) — NEGATIVE adjustments:
   - Any refactoring not strictly required for the fix: -5 to -10
   - Refactoring that adds complexity or changes structure unnecessarily: -8 to -15
   - Refactoring that makes it harder to verify the fix is correct: -10 to -15
   - Refactoring that suggests the patch is cosmetic rather than substantive: -15 to -25

   EXCEPTION (rare) — NEUTRAL (0 adjustment):
   - Refactoring that is PROVABLY REQUIRED for the fix to compile/work (e.g., must add parameter to pass data)
   - You must cite specific evidence why the fix cannot work without this refactoring

   NEVER give positive adjustments for refactoring. Clean code is not a bonus in bug-fix patches.

Decision process:
   --- Fix evaluation (base score) ---
   a. Start at 50 (uncertain). Add points only for concrete fix evidence; subtract for concerns.
   
   --- Refactoring penalty (default negative) ---
   b. For each refactoring listed:
      - DEFAULT: Apply -5 to -15 penalty (refactoring adds risk)
      - ONLY IF you can prove it's required for the fix: 0 (no penalty, but no bonus)
   c. Sum up the penalties
   
   --- Final score ---
   d. final_score = base_score + refactoring_adjustment (clamped to 0-100)

Output requirements:
- Respond ONLY with a fenced JSON block (```json ... ```).
- Include the fields:
  * "resolution_score": integer 0-100 indicating likelihood of resolving the issue
  * "patch_resolution": "resolves" (score >= 50) or "does_not_resolve" (score < 50)
  * "confidence": "low", "medium", or "high"
  * "reasoning": structured explanation showing base score, each refactoring's penalty (with justification), and final score
""".strip()


SELECT_JUDGEMENT_USER_PROMPT_WITHOUT_REFACTORING = """
You will be provided with an issue description, optional code context, and the candidate patch diff.

<issue>
{issue_description}
</issue>

<code>
{code_sections}
</code>

<patch>
{patch_diff}
</patch>

Evaluation task:
1. Start from a NEUTRAL baseline of 50 (uncertain). Be skeptical.
2. Adjust the score based on concrete evidence:
   - Clear evidence the patch addresses the root cause: +15 to +30
   - Fix logic is complete and covers edge cases: +10 to +20
   - Uncertainty about whether the fix is correct: -5 to -15
   - Obvious bugs, regressions, or incomplete handling: -15 to -30
   - Patch does not seem to address the issue at all: -20 to -40
3. Provide both the numerical score and binary verdict (resolves if score >= 50).

Be conservative: only give high scores (80+) when the fix is clearly correct and complete.

Respond strictly with the JSON format specified in the system prompt.
""".strip()


SELECT_JUDGEMENT_USER_PROMPT_WITH_REFACTORING = """
You will be provided with an issue description, optional code context, the candidate patch diff, and refactoring context that lists detected refactorings (structural changes beyond the minimal fix).

<issue>
{issue_description}
</issue>

<code>
{code_sections}
</code>

<patch>
{patch_diff}
</patch>

<refactorings>
{refactoring_section}
</refactorings>

Evaluation task:
1. Assess the BASE fix score starting from 50 (neutral/uncertain):
   - Clear evidence the patch addresses the root cause: +15 to +30
   - Fix logic is complete and covers edge cases: +10 to +20
   - Uncertainty about whether the fix is correct: -5 to -15
   - Obvious bugs, regressions, or incomplete handling: -15 to -30

2. Apply PENALTIES for each refactoring (refactoring in bug-fix = risk):
   
   DEFAULT — Apply penalty:
   - Any refactoring not strictly required for the fix: -5 to -10
   - Refactoring that adds unnecessary complexity: -8 to -15
   - Refactoring that obscures the fix or makes verification harder: -10 to -15
   - Refactoring suggesting the patch is cosmetic, not substantive: -15 to -25
   
   EXCEPTION — No penalty (0), only if you can PROVE:
   - The fix literally cannot compile or work without this specific change
   - Cite the exact reason why this refactoring is unavoidable

   NEVER give bonuses for refactoring. This is a bug-fix, not a cleanup task.

3. Calculate final score:
   final_score = base_score + sum(refactoring_penalties)
   
4. Verdict: "resolves" if final_score >= 50, "does_not_resolve" otherwise

Respond strictly with the JSON format specified in the system prompt.
""".strip()


JUDGEMENT_EXAMPLE_JSON = """
{
  "patch_resolution": "does_not_resolve",
  "confidence": "medium",
  "reasoning": "Refactoring validation: <per-ref summary verdicts>. Fix evidence: <diff hunks that fail to address the issue>."
}
""".strip()


@dataclass(frozen=True)
class CodeSnippet:
    """Represents a code excerpt that should be embedded in the prompt."""

    file_path: str
    content: str

    def render(self) -> str:
        return f"[start of {self.file_path}]\n{self.content}\n[end of {self.file_path}]"


@dataclass(frozen=True)
class PromptMessages:
    """Container for the three-part prompt payload consumed by an LLM."""

    global_system: str
    system: str
    user: str
    example_json: Optional[str] = None


def _normalise_multiline(value: Optional[Iterable[str] | str]) -> str:
    """Convert an iterable of strings or a plain string into a formatted block."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return "\n".join(fragment.rstrip() for fragment in value).strip()


def _render_code_sections(snippets: Sequence[CodeSnippet]) -> str:
    if not snippets:
        return "[no code context provided]"
    return "\n\n".join(snippet.render() for snippet in snippets)


@dataclass(frozen=True)
class PromptTemplatePack:
    """Aggregate of system and user prompt templates for selection experiment."""

    global_system_prompt: str = GLOBAL_SYSTEM_PROMPT
    # Select-task judgement prompts (more lenient)
    select_judgement_system_prompt_without_refactoring: str = SELECT_JUDGEMENT_SYSTEM_PROMPT_WITHOUT_REFACTORING
    select_judgement_system_prompt_with_refactoring: str = SELECT_JUDGEMENT_SYSTEM_PROMPT_WITH_REFACTORING
    select_judgement_user_template_without_refactoring: str = SELECT_JUDGEMENT_USER_PROMPT_WITHOUT_REFACTORING
    select_judgement_user_template_with_refactoring: str = SELECT_JUDGEMENT_USER_PROMPT_WITH_REFACTORING

    def build_select_judgement_prompt(
        self,
        *,
        issue_description: str,
        code_snippets: Sequence[CodeSnippet] | None,
        patch_diff: str,
        refactoring_context: str | Iterable[str] | None = None,
        include_refactoring_context: bool = False,
    ) -> PromptMessages:
        """Compose the LLM-as-a-judge messages for patch selection task.
        
        This uses the LENIENT judgement prompts optimized for finding
        patches that resolve the issue (resolves as the desired outcome).
        Refactoring context is treated as informational, not blocking.
        """

        ref_section = ""
        if include_refactoring_context:
            ref_payload = _normalise_multiline(refactoring_context)
            if not ref_payload:
                ref_payload = "[no refactoring context provided]"
            ref_section = f"\n<refactoring_context>\n{ref_payload}\n</refactoring_context>\n"

        user_template = (
            self.select_judgement_user_template_with_refactoring
            if include_refactoring_context
            else self.select_judgement_user_template_without_refactoring
        )

        # Build format args - only include refactoring_section for with_refactoring template
        format_args = {
            "issue_description": issue_description.strip(),
            "code_sections": _render_code_sections(code_snippets or ()),
            "patch_diff": _normalise_multiline(patch_diff),
        }
        if include_refactoring_context:
            format_args["refactoring_section"] = ref_section

        user_prompt = user_template.format(**format_args)

        system_prompt = (
            self.select_judgement_system_prompt_with_refactoring
            if include_refactoring_context
            else self.select_judgement_system_prompt_without_refactoring
        )

        return PromptMessages(
            global_system=self.global_system_prompt,
            system=system_prompt,
            user=user_prompt,
            example_json=JUDGEMENT_EXAMPLE_JSON,
        )


__all__ = [
    "CodeSnippet",
    "PromptMessages",
    "PromptTemplatePack",
    "GLOBAL_SYSTEM_PROMPT",
    "SELECT_JUDGEMENT_SYSTEM_PROMPT_WITHOUT_REFACTORING",
    "SELECT_JUDGEMENT_SYSTEM_PROMPT_WITH_REFACTORING",
    "SELECT_JUDGEMENT_USER_PROMPT_WITHOUT_REFACTORING",
    "SELECT_JUDGEMENT_USER_PROMPT_WITH_REFACTORING",
    "JUDGEMENT_EXAMPLE_JSON",
]
