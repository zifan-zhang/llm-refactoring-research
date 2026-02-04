"""Prompt templates for the patch refinement experiment.

This module contains the prompts used by 4_run_refinement_experiment_refine.py
to generate refined patches from assessment results. It is extracted from
prompt_template_pack.py for clarity and focused usage.

The refinement task applies remove/fix actions to refactorings in bug-fix patches
while preserving the core fix functionality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


# =============================================================================
# Global system prompt (shared across all RQ3 tasks)
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
# Patch refinement prompts (Apply refactoring actions to produce refined patch)
# =============================================================================
# These prompts are used in the refinement step to modify patches based on
# the assessment results. The LLM is instructed to apply remove/fix actions.

REFINEMENT_SYSTEM_PROMPT = """
You are a senior Java engineer tasked with refining a bug-fix patch. Your goal is to apply the specified actions to each refactoring while keeping the core bug fix intact.

Key principles:

1. **Preserve the bug fix**: The primary fix must remain functional after refinement
2. **Apply actions precisely**:
   - KEEP: Leave the refactoring changes as-is
   - REMOVE: Revert only the refactoring-related changes, preserving the bug fix
   - FIX: Apply the fix_suggestion to make the refactoring safe
3. **Minimize changes**: Only modify what is necessary to apply the actions
4. **Maintain compilability**: Ensure the refined patch will compile successfully

Output requirements:
- Output ONLY a unified diff for the refined patch
- Use standard Git diff format with the following structure:
  * diff --git a/path b/path
  * (optional) index line
  * --- a/path
  * +++ b/path
  * @@ ... @@ hunks
- Each file change must include at minimum: diff --git, --- a/path, +++ b/path lines
- Wrap the entire diff in a ```diff code block
- Do NOT include explanations outside the code block
- The output must be directly applicable using "git apply" or "patch" commands
""".strip()


REFINEMENT_USER_PROMPT = """
You will be given:
1) The original patch diff
2) A list of refactoring assessments with recommended actions

<patch>
{patch_diff}
</patch>

<assessments>
{refactoring_assessments}
</assessments>

Apply the recommended actions:

For each refactoring assessment:
- If action is "keep": leave those changes unchanged
- If action is "remove": revert the refactoring-related changes while preserving the bug fix
- If action is "fix": apply the fix_suggestion to correct the safety issues

Important:
- The refined patch must still fix the original bug
- Output must be a complete Git unified diff format (with "diff --git", "index", "---", "+++" headers)
- Wrap the entire diff in ```diff ... ```
- Ensure the patch is syntactically correct and compilable
- The output must be directly applicable using "git apply" command
""".strip()


REFINEMENT_EXAMPLE_OUTPUT = """
```diff
diff --git a/src/main/java/com/example/Foo.java b/src/main/java/com/example/Foo.java
index abc1234..def5678 100644
--- a/src/main/java/com/example/Foo.java
+++ b/src/main/java/com/example/Foo.java
@@ -10,7 +10,7 @@
     public void processData(String data) {
-        if (data == null) {
+        if (data == null || data.isEmpty()) {
             throw new IllegalArgumentException("data cannot be null");
         }
         // ... rest of the method
     }
```
""".strip()


# =============================================================================
# Helper utilities and data structures
# =============================================================================

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


# =============================================================================
# Prompt builder
# =============================================================================

def build_patch_refinement_prompt(
    *,
    patch_diff: str,
    refactoring_assessments: str | Iterable[str],
) -> PromptMessages:
    """Compose the LLM messages for patch refinement task.
    
    This uses the REFINEMENT prompts to modify patches based on
    assessment results, applying remove/fix actions while preserving
    the core bug fix.
    
    Args:
        patch_diff: The original patch diff to refine
        refactoring_assessments: Assessment results with actions (as JSON string)
    
    Returns:
        PromptMessages object with global system, system, and user prompts
    
    Example:
        >>> import json
        >>> assessments = [
        ...     {
        ...         "refactoring_type": "Rename Variable",
        ...         "action": "remove",
        ...         "reasoning": "Cosmetic change"
        ...     }
        ... ]
        >>> prompt = build_patch_refinement_prompt(
        ...     patch_diff="--- a/Foo.java\\n+++ b/Foo.java\\n...",
        ...     refactoring_assessments=json.dumps(assessments, indent=2)
        ... )
        >>> # Use prompt.global_system, prompt.system, prompt.user with LLM API
    """
    # Normalize assessments to string
    assessments_payload = _normalise_multiline(refactoring_assessments)
    if not assessments_payload:
        assessments_payload = "[]"
    
    # Build user prompt
    user_prompt = REFINEMENT_USER_PROMPT.format(
        patch_diff=_normalise_multiline(patch_diff),
        refactoring_assessments=assessments_payload,
    )
    
    return PromptMessages(
        global_system=GLOBAL_SYSTEM_PROMPT,
        system=REFINEMENT_SYSTEM_PROMPT,
        user=user_prompt,
        example_json=REFINEMENT_EXAMPLE_OUTPUT,
    )


__all__ = [
    "PromptMessages",
    "GLOBAL_SYSTEM_PROMPT",
    "REFINEMENT_SYSTEM_PROMPT",
    "REFINEMENT_USER_PROMPT",
    "REFINEMENT_EXAMPLE_OUTPUT",
    "build_patch_refinement_prompt",
]
