# AI Tool Comparison (the assignment requires at least 2 tools on one task)

**Task:** Held-Karp dynamic programming for an exact travelling-salesman tour (`docs/tool_comparison/TASK_PROMPT.md`). It is a Chapter 6 discrete-optimization method that is not used elsewhere in this repository, so neither tool can copy an existing answer from the project.
**Same conditions for every tool:** the identical prompt pasted into a fresh chat; the first answer is saved unedited; any follow-up prompt is counted as an iteration.
**Scoring is automatic:** `python docs/tool_comparison/check.py <file>` checks 60 instances against brute force (n = 1-8, symmetric and asymmetric), tour validity, and runtime at n = 13 and 15.

| Criterion | Tool 1: Claude Code (Claude Opus 5.5) | Tool 2: _name + model version_ |
|---|---|---|
| Correctness (checker) | 60/60 correct | _fill in_ |
| Runtime n=13 / n=15 | 0.06 s / 0.48 s | _fill in_ |
| Hallucinations (invented APIs, wrong complexity claims, missing edge cases) | none found; handles n=1 and asymmetric distances | _fill in_ |
| Prompt iterations needed to reach 60/60 | 1 | _fill in_ |
| Time to working code | one response | _fill in_ |
| Readability / explainability | about 35 lines, bitmask DP with a parent table; comments explain the state | _fill in_ |

Files: `docs/tool_comparison/claude_code_solution.py` (Tool 1, unedited first attempt), `docs/tool_comparison/<tool2>_solution.py` (Tool 2).

**Caveat on Tool 1:** this answer was produced late in a long session in which the same tool had already written many optimization algorithms, so it is not a cold-start test. Tool 2 should be given the prompt in a fresh chat; note this difference in the conclusion.

## How to complete Tool 2 (about 5 minutes)
1. Open ChatGPT (or Gemini, Copilot, ...) in a new chat and paste the text of `TASK_PROMPT.md`.
2. Save its code unedited as `docs/tool_comparison/chatgpt_solution.py`.
3. Run `python docs/tool_comparison/check.py docs/tool_comparison/chatgpt_solution.py`.
4. If it is not 60/60, reply to the tool with the checker's error lines, save the new version, re-run, and count the iterations.
5. Fill in the right-hand column and the conclusion; add the tool to the AI tools register in `README.md`; save a screenshot of the chat in `prompts/`.

## Conclusion
_Write after filling in Tool 2: which tool was correct first time, which needed fixes, which made unsupported claims, and which code you could explain more easily in the viva._
