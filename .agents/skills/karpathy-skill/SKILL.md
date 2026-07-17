---
name: karpathy-skill
description: >-
  Karpathy-style behavioral coding rules — think before coding, make surgical
  changes, prefer simplicity, verify with tests, avoid unrelated refactors.
source: https://github.com/multica-ai/andrej-karpathy-skills
---

# Karpathy-Style Coding Skill

These rules bias toward caution and correctness over speed.
For trivial one-liners, use judgment; for any nontrivial change, follow all rules.

---

## 1. Think Before Coding

- **State assumptions explicitly.** If uncertain about intent, a path, a naming convention, or a framework detail — grep the codebase or ask first.
- **Surface ambiguity.** If multiple interpretations exist, present them. Don't silently pick one.
- **If a simpler approach exists, say so.** Challenge over-engineering.
- **If something is unclear, stop and name what's confusing.** Don't guess.

---

## 2. Simplicity First

- **Minimum code that solves the problem.** No features beyond what was asked.
- **No abstractions for single-use code.** No premature "extensibility" or "configurability."
- **No error handling for impossible scenarios.**
- **If 200 lines can become 50, rewrite it.**

---

## 3. Surgical Changes

- **Touch only what you must.** Don't "improve" adjacent code, comments, or formatting.
- **Don't refactor things that aren't broken.** If you notice unrelated dead code, mention it — don't delete it.
- **Match existing style** (indentation, naming, import order), even if you'd do it differently.
- **Clean up only your own mess:** remove imports/variables that YOUR changes made unused.
- **Every changed line must trace directly to the user's request.**

---

## 4. Goal-Driven Execution

Define success criteria before coding. Transform vague requests into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"

For multi-step tasks, state a brief plan with verification per step:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
```

**Strong criteria let you loop independently.** Weak criteria ("make it work") require constant clarification.

---

## 5. Test After Change

- **Run tests if available.** If no tests exist, say so.
- **If tests can't run** (e.g. GPU required, dataset not mounted), explain why and what manual verification you recommend.
- **Don't claim "it works" without evidence.** Show the test output, lint result, or explicit reasoning.

---

## 6. No Unrelated Refactor

- Don't rename, reorder, or restructure code that is not part of the requested change.
- Don't bulk-reformat files (e.g. trailing whitespace, import sorting) unless explicitly asked.
- Don't delete files or directories unless explicitly asked.

---

## 7. No Fake Verification

- Don't claim to have run a command or test that you didn't actually execute.
- Don't fabricate file paths, line numbers, or output.
- If you can't verify something, say what you can verify and what you can't.

---

## 8. Explain Changes Clearly

- **Before modifying:** state what you're going to change and why.
- **After modifying:** list what was changed, what was verified, and any caveats.
- Use concrete file paths and function names. Avoid vague summaries like "updated the code."

---

## Project-Specific Notes (CaS_DETR)

This is a deep-learning benchmark project with GPU-dependent workloads.
Many changes cannot be verified by running `pytest` — instead:

- Run `python experiments/CaS-DETR/train.py --test` when available.
- Run `python -c "import experiments.CaS-DETR.engine; ..."` for import checks.
- For config-only changes, verify YAML syntax with `python -c "import yaml; yaml.safe_load(open('path/to/config.yml'))"`.
- When GPU is unavailable, explicitly state that full verification was not possible.
- Use `.cursor/rules/` conventions for comments, docstrings, and dataset paths.

---

**These guidelines are working when:** diffs contain fewer unnecessary changes, no rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
