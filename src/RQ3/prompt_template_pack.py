"""Prompt templates for patch judgement experiment.

This module provides the prompt templates and building utilities specifically
for the patch judgement experiment (1_run_patch_judgement_experiment.py).

The experiment compares two LLM-as-a-judge conditions:
1. Baseline: judge sees only the issue description and patch diff.
2. Refactoring-aware: judge additionally receives detected refactoring context.
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


JUDGEMENT_SYSTEM_PROMPT_WITHOUT_REFACTORING = """
You are an impartial senior Java engineer acting as a correctness judge. Your goal is to determine whether a candidate patch resolves the reported issue when only code context and the diff are available.

Evaluation principles:
1. Focus on the functional intent described in the issue.
2. Use only the provided code context and diff;
3. Treat "does_not_resolve" as the positive label: only answer "resolves" when the diff conclusively fixes the issue without adding new risks.
4. Identify concrete diff evidence that fixes the defect. If none, answer "does_not_resolve".

Output requirements:
- Respond ONLY with a fenced JSON block (```json ... ```).
- Include the fields:
  * "patch_resolution": "resolves" or "does_not_resolve"
  * "confidence": "low", "medium", or "high"
  * "reasoning": structured explanation that lists the fix evidence, highlights behavioural checks, and notes remaining risks
""".strip()


JUDGEMENT_SYSTEM_PROMPT_WITH_REFACTORING = """
You are an impartial senior Java engineer acting as a correctness judge. Your goal is to decide whether a candidate patch resolves the reported issue while also considering refactoring context that explains structural changes.

Evaluation principles:
1. Focus on the functional intent described in the issue.
2. Use the code context, diff, and refactoring summaries to interpret structural changes. Assume the listed refactorings are unintended tangles mixed into the bug-fix commit and therefore HIGH RISK by default until you find concrete diff evidence that proves they are safe.
3. Treat "does_not_resolve" as the positive label: only answer "resolves" once both the bug fix and every referenced refactoring are proven safe.
4. Refactoring summaries are NOT signals of a good patch; they are warning flags that must be justified by explicit fix evidence.
5. Method-level or class-level refactorings are high risk unless the diff proves that behaviour is preserved, recompilation succeeds, and API contracts stay intact.
6. Refactoring summaries are hypotheses, not evidence. Never rely on them alone—locate the corresponding diff hunk and confirm wiring (e.g., new parameters passed through every caller) before accepting a change.
7. Follow this decision process:
   --- Refactoring validation first ---
   a. For each refactoring summary, cite the exact diff hunk (file + method) that matches it and judge whether behaviour and compilation are preserved. Missing or unsafe mappings force "does_not_resolve".
   b. Ensure every refactoring-introduced API surface (parameters, fields, factories, wiring) is updated across the diff. Caller gaps or ambiguous wiring force "does_not_resolve".
   c. Treat every unresolved or unmapped refactoring summary as a blocker. Missing evidence, unverifiable descriptions, or ambiguous intent force "does_not_resolve".

   --- Fix validation after refactoring clears ---
   d. Identify concrete diff evidence that fixes the defect. If none, answer "does_not_resolve".
   e. Only after both the refactorings and the fix are proven safe may you answer "resolves"—cite the fix evidence plus the per-refactoring verdicts.

Output requirements:
- Respond ONLY with a fenced JSON block (```json ... ```).
- Include the fields:
  * "patch_resolution": "resolves" or "does_not_resolve"
  * "confidence": "low", "medium", or "high"
  * "reasoning": structured explanation referencing fix evidence, each refactoring's status (mapped location + safety verdict), and any remaining concerns
""".strip()


JUDGEMENT_USER_PROMPT_WITHOUT_REFACTORING = """
You will be provided with an issue description, optional code context, the candidate patch diff, and optionally refactoring context.

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
1. Decide whether the patch resolves the issue as described.
2. Highlight the key behavioural changes that support or refute the fix.
3. Identify any evidence of regressions or incomplete handling.
4. Treat "does_not_resolve" as the positive label—flagging unresolved patches is the primary goal, and you should only emit "resolves" when the diff conclusively proves the bug is fixed.
5. Estimate your confidence (low/medium/high) based on the available information.

Respond strictly with the JSON format specified in the system prompt.
""".strip()


JUDGEMENT_USER_PROMPT_WITH_REFACTORING = """
You will be provided with an issue description, optional code context, the candidate patch diff, and refactoring context that lists the detected refactorings.

<issue>
{issue_description}
</issue>

<code>
{code_sections}
</code>

<patch>
{patch_diff}
</patch>
{refactoring_section}
Evaluation task:
1. Decide whether the patch resolves the issue as described.
2. Highlight the key behavioural changes that support or refute the fix.
3. Identify any evidence of regressions or incomplete handling.
4. For each refactoring summary, cite the exact diff hunk (file + function/method) that matches it and judge whether it is safe. If you cannot locate the referenced code or prove it is safe, the final verdict MUST be "does_not_resolve".
5. Treat "does_not_resolve" as the positive label—flagging unresolved patches is the primary goal, and you should only emit "resolves" when the diff conclusively proves the bug is fixed and every referenced refactoring is safe.
6. Structure your reasoning with separate sections for refactoring validation (one entry per summary) and fix evidence, then estimate your confidence (low/medium/high).

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
    """Render a sequence of code snippets into a formatted block."""
    if not snippets:
        return "[no code context provided]"
    return "\n\n".join(snippet.render() for snippet in snippets)


@dataclass(frozen=True)
class PromptTemplatePack:
    """Prompt template pack for patch judgement experiment."""

    global_system_prompt: str = GLOBAL_SYSTEM_PROMPT
    judgement_system_prompt_without_refactoring: str = JUDGEMENT_SYSTEM_PROMPT_WITHOUT_REFACTORING
    judgement_system_prompt_with_refactoring: str = JUDGEMENT_SYSTEM_PROMPT_WITH_REFACTORING
    judgement_user_template_without_refactoring: str = JUDGEMENT_USER_PROMPT_WITHOUT_REFACTORING
    judgement_user_template_with_refactoring: str = JUDGEMENT_USER_PROMPT_WITH_REFACTORING

    def build_patch_judgement_prompt(
        self,
        *,
        issue_description: str,
        code_snippets: Sequence[CodeSnippet] | None,
        patch_diff: str,
        refactoring_context: str | Iterable[str] | None = None,
        include_refactoring_context: bool = False,
    ) -> PromptMessages:
        """Compose the LLM-as-a-judge messages for patch validation.
        
        This uses the STRICT judgement prompts optimized for detecting
        unresolved patches (does_not_resolve as positive class).
        
        Args:
            issue_description: The bug report or issue description
            code_snippets: Optional code context snippets
            patch_diff: The patch diff to evaluate
            refactoring_context: Optional refactoring context (only used if include_refactoring_context=True)
            include_refactoring_context: Whether to include refactoring context in the prompt
        
        Returns:
            PromptMessages object with global system, system, and user prompts
        """

        ref_section = ""
        if include_refactoring_context:
            ref_payload = _normalise_multiline(refactoring_context)
            if not ref_payload:
                ref_payload = "[no refactoring context provided]"
            ref_section = f"\n<refactoring_context>\n{ref_payload}\n</refactoring_context>\n"

        user_template = (
            self.judgement_user_template_with_refactoring
            if include_refactoring_context
            else self.judgement_user_template_without_refactoring
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
            self.judgement_system_prompt_with_refactoring
            if include_refactoring_context
            else self.judgement_system_prompt_without_refactoring
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
    "JUDGEMENT_SYSTEM_PROMPT_WITHOUT_REFACTORING",
    "JUDGEMENT_SYSTEM_PROMPT_WITH_REFACTORING",
    "JUDGEMENT_USER_PROMPT_WITHOUT_REFACTORING",
    "JUDGEMENT_USER_PROMPT_WITH_REFACTORING",
    "JUDGEMENT_EXAMPLE_JSON",
]
