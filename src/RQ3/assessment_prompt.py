"""Prompt templates for the Refactoring Assessment Experiment.

This module contains the prompts used in the refactoring assessment workflow
(3_run_refinement_experiment_assessment.py). It evaluates the necessity and
safety of detected refactorings in bug-fix patches and recommends appropriate
actions (keep/remove/fix).

Extracted from: RQ3/prompt_template_pack.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


# =============================================================================
# Global System Prompt
# =============================================================================

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


# =============================================================================
# Refactoring Assessment Prompts
# =============================================================================

ASSESSMENT_SYSTEM_PROMPT = """
You are a senior Java engineer specializing in patch analysis and refactoring evaluation. Your goal is to assess each refactoring detected in a bug-fix patch and recommend appropriate actions.

Evaluation Dimensions:

1. **Necessity** - Is this refactoring required for the fix to work?
   - REQUIRED: The fix cannot work without this refactoring (e.g., must add parameter to pass critical data)
   - UNNECESSARY: The refactoring is cosmetic cleanup unrelated to the fix

2. **Safety** - Is the refactoring implementation correct?
   - SAFE: The refactoring is correctly implemented with no issues
     * All API changes are consistently applied across all call sites
     * Type signatures are compatible or properly updated
     * Behavior is preserved (same logic, proper wiring, equivalent control flow)
   
   - UNSAFE: The refactoring has implementation problems (specify what)
     * Missing updates at some call sites
     * Type incompatibilities or compilation errors
     * Behavior changes or incorrect wiring
     * Breaking changes to public APIs

3. **Recommended Action** - What should be done with this refactoring?
   - KEEP: Required + safe → keep as-is
   - REMOVE: Unnecessary (even if safe) → remove entirely to minimize patch
   - FIX: Required but unsafe → fix the safety issues while preserving functionality

Decision Process:

For EACH detected refactoring, evaluate:
a. Locate the exact diff hunks that implement this refactoring
b. Assess necessity: Is it required/unnecessary for the bug fix?
c. Check safety: Are there any implementation issues?
d. If unsafe, identify specific safety issues (missing updates, type errors, etc.)
e. Recommend action based on necessity + safety combination:
   
   Necessity     Safety    → Action
   ─────────────────────────────────
   REQUIRED      SAFE      → KEEP
   REQUIRED      UNSAFE    → FIX (must preserve fix functionality)
   UNNECESSARY   SAFE      → REMOVE (minimize patch)
   UNNECESSARY   UNSAFE    → REMOVE (no value, has risks)

f. If action is FIX, provide specific suggestions on how to address the issues

Consistency rules:
- If you mark safety as "unsafe", action MUST be "fix" (if required) or "remove" (if unnecessary).
- If you mark safety as "safe", safety_issues MUST be empty.
- If action is "keep" or "remove", fix_suggestion MUST be null.
- If you cannot find the refactoring in the diff, mark it as unnecessary + unsafe and explain the mismatch in safety_issues.
- Do NOT claim a refactoring is safe while listing safety issues or requesting a fix.
- Do NOT default to "has_issues" if the evidence shows the refactoring is required and implemented safely.

Overall verdict rules:
- "all_safe" only if every refactoring is required+safe with action "keep" and no safety issues.
- "has_issues" if any refactoring is unsafe or unnecessary (action fix/remove).
- "uncertain" only if the diff is insufficient to verify refactorings or safety.

Output requirements:
- Respond ONLY with a fenced JSON block (```json ... ```).
- Include the fields:
  * "overall_verdict": "all_safe" | "has_issues" | "uncertain"
  * "confidence": "low" | "medium" | "high"
  * "refactoring_assessments": array of per-refactoring evaluations (see below)
  * "summary": brief summary of findings
  * "actions_needed": {"keep": N, "remove": N, "fix": N}

Per-refactoring assessment structure:
{
  "refactoring_type": "Extract Method",
  "location": "file.java:MethodName",
  "necessity": "required" | "unnecessary",
  "safety": "safe" | "unsafe",
  "safety_issues": ["issue 1", "issue 2"],  // empty if safe
  "action": "keep" | "remove" | "fix",
  "fix_suggestion": "How to fix if action is fix",  // null if not applicable
  "reasoning": "detailed explanation of assessment"
}
""".strip()


ASSESSMENT_USER_PROMPT = """
You will be provided with an issue description, optional code context, the candidate patch diff, and a list of detected refactorings.

<issue>
{issue_description}
</issue>

<code>
{code_sections}
</code>

<patch>
{patch_diff}
</patch>

<detected_refactorings>
{refactoring_list}
</detected_refactorings>

Evaluation task:

1. For EACH refactoring in the list above:
   a. Locate the corresponding diff hunks (cite file names and line numbers)
   b. Assess NECESSITY: Is it required/unnecessary for the bug fix?
   c. Check SAFETY: Are there any implementation issues?
      - Missing call site updates?
      - Type incompatibilities?
      - Behavior changes?
   d. If unsafe, list specific safety issues
   e. Recommend ACTION based on the necessity × safety matrix:
      - Required + Safe → KEEP
      - Required + Unsafe → FIX (must preserve functionality)
      - Unnecessary + Safe/Unsafe → REMOVE (minimize patch)
   f. If action is FIX, suggest how to address the issues
   g. If the refactoring cannot be located in the diff, mark it as unnecessary + unsafe and explain the mismatch

2. Provide overall assessment:
   - "overall_verdict": "all_safe" (all are safe and necessary) | "has_issues" (some need action) | "uncertain"
   - "actions_needed": Count how many refactorings need each action type

3. Be OBJECTIVE:
   - Necessary refactorings with implementation issues should be marked for FIX, not REMOVE
   - Unnecessary refactorings should be marked for REMOVE, even if they are safe
   - Focus on minimizing the patch while preserving the fix functionality
   - Keep reasoning consistent with safety/action (no contradictions)
   - If a refactoring is fully supported by the diff with no issues, mark it safe and allow "all_safe"

Respond strictly with the JSON format specified in the system prompt.
""".strip()


ASSESSMENT_EXAMPLE_JSON = """
{
  "overall_verdict": "has_issues",
  "confidence": "high",
  "refactoring_assessments": [
    {
      "refactoring_type": "Add Parameter",
      "location": "src/main/Foo.java:processData",
      "necessity": "required",
      "safety": "unsafe",
      "safety_issues": [
        "Missing updates at Bar.java:45 call site",
        "Missing updates at Baz.java:78 call site"
      ],
      "action": "fix",
      "fix_suggestion": "Update all 3 call sites to pass the new 'config' parameter. Add parameter with proper null checks.",
      "reasoning": "The new parameter is required to pass configuration data for the fix, but 2 out of 3 call sites were not updated, which will cause compilation errors."
    },
    {
      "refactoring_type": "Rename Variable",
      "location": "src/main/Foo.java:processData",
      "necessity": "unnecessary",
      "safety": "safe",
      "safety_issues": [],
      "action": "remove",
      "fix_suggestion": null,
      "reasoning": "Variable renaming from 'data' to 'inputData' is cosmetic cleanup unrelated to the bug fix. Should be removed to minimize the patch."
    }
  ],
  "summary": "2 refactorings detected: 1 required but unsafe (needs fixing), 1 unnecessary (should be removed).",
  "actions_needed": {
    "keep": 0,
    "remove": 1,
    "fix": 1
  }
}
""".strip()


# =============================================================================
# Data Classes
# =============================================================================

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


# =============================================================================
# Helper Functions
# =============================================================================

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


# =============================================================================
# Prompt Builder Class
# =============================================================================

@dataclass(frozen=True)
class AssessmentPromptPack:
    """Prompt template pack for refactoring assessment workflow.
    
    This simplified class focuses only on the assessment prompts used
    to evaluate detected refactorings in bug-fix patches.
    """

    global_system_prompt: str = GLOBAL_SYSTEM_PROMPT
    assessment_system_prompt: str = ASSESSMENT_SYSTEM_PROMPT
    assessment_user_template: str = ASSESSMENT_USER_PROMPT

    def build_assessment_prompt(
        self,
        *,
        issue_description: str,
        code_snippets: Sequence[CodeSnippet] | None,
        patch_diff: str,
        refactoring_list: str | Iterable[str],
    ) -> PromptMessages:
        """Compose the LLM-as-a-judge messages for refactoring assessment.
        
        This uses the ASSESSMENT prompts to evaluate detected refactorings
        in a bug-fix patch along two dimensions (necessity and safety),
        and recommend appropriate actions (keep/remove/fix).
        
        Args:
            issue_description: The bug report or issue description
            code_snippets: Optional code context snippets
            patch_diff: The patch diff to evaluate
            refactoring_list: List of detected refactorings (as string or iterable)
        
        Returns:
            PromptMessages object with global system, system, and user prompts
        """
        # Normalize refactoring list
        ref_payload = _normalise_multiline(refactoring_list)
        if not ref_payload:
            ref_payload = "[no refactorings detected]"
        
        # Build user prompt
        user_prompt = self.assessment_user_template.format(
            issue_description=issue_description.strip(),
            code_sections=_render_code_sections(code_snippets or ()),
            patch_diff=_normalise_multiline(patch_diff),
            refactoring_list=ref_payload,
        )
        
        return PromptMessages(
            global_system=self.global_system_prompt,
            system=self.assessment_system_prompt,
            user=user_prompt,
            example_json=ASSESSMENT_EXAMPLE_JSON,
        )


__all__ = [
    "CodeSnippet",
    "PromptMessages",
    "AssessmentPromptPack",
    "GLOBAL_SYSTEM_PROMPT",
    "ASSESSMENT_SYSTEM_PROMPT",
    "ASSESSMENT_USER_PROMPT",
    "ASSESSMENT_EXAMPLE_JSON",
]
