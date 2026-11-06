# AI Tool Comparison (the assignment requires at least 2 tools on one task)

**Task:** Held-Karp dynamic programming for an exact travelling-salesman tour (`docs/tool_comparison/TASK_PROMPT.md`). It is a Chapter 6 discrete-optimization method that is not used elsewhere in this repository, so neither tool can copy an existing answer from the project.
**Same conditions for every tool:** the identical prompt pasted into a fresh chat; the first answer is saved unedited; any follow-up prompt is counted as an iteration.
**Scoring is automatic:** `python docs/tool_comparison/check.py <file>` checks 60 instances against brute force (n = 1-8, symmetric and asymmetric), tour validity, and runtime at n = 13 and 15.

| Criterion | Tool 1: Claude Code (Claude Opus 5.5, `claude-opus-5-5`) | Tool 2: Claude Haiku 5.5 (`claude-haiku-5-5`), fresh agent |
|---|---|---|
| Correctness (checker) | 60/60 correct | 60/60 correct |
| Runtime n=13 / n=15 | 0.06 s / 0.48 s | 0.05 s / **0.27 s** |
| Hallucinations (invented APIs, wrong complexity claims, missing edge cases) | none found; handles n=1 and asymmetric distances | none found; claimed O(n^2 2^n) time and O(n 2^n) memory (correct); also guards n=0 |
| Prompt iterations needed to reach 60/60 | 1 | 1 |
| Time to working code | one response | one response, 12 s of agent time, 2 tool calls |
| Readability / explainability | 35 lines, bitmask DP with a parent table; terse | 68 lines, same algorithm, more comments and explicit loops; slightly easier to explain |

Files: `docs/tool_comparison/claude_code_solution.py` (Tool 1) and `docs/tool_comparison/claude_haiku_solution.py` (Tool 2), both unedited first attempts.

**How Tool 2 was run:** Claude Code launched a separate agent on the Claude Haiku 5.5 model with only the task prompt, instructed not to read any repository file and not to run or test its code; its file was then scored by the same checker.
**Caveat on Tool 1:** this answer was produced late in a long session in which the same tool had already written many optimization algorithms, so it is not a cold-start test; Tool 2 was.
**Caveat on the comparison:** both are Claude models from one vendor, used through Claude Code. This compares two models, not two products; a comparison with a different product (ChatGPT, Gemini) can be added with the steps below.

## Optional: add a third tool from another vendor (about 5 minutes)
1. Open ChatGPT (or Gemini, Copilot, ...) in a new chat and paste the text of `TASK_PROMPT.md`.
2. Save its code unedited as `docs/tool_comparison/chatgpt_solution.py`.
3. Run `python docs/tool_comparison/check.py docs/tool_comparison/chatgpt_solution.py`.
4. If it is not 60/60, reply to the tool with the checker's error lines, save the new version, re-run, and count the iterations.
5. Fill in the right-hand column and the conclusion; add the tool to the AI tools register in `README.md`; save a screenshot of the chat in `prompts/`.

## Conclusion
Both models solved the task correctly on the first attempt (60/60 against brute force, valid tours, asymmetric distances and n = 1 handled) and neither invented an API or made a false complexity claim. The smaller model was **1.8x faster at n = 15** (0.27 s vs 0.48 s) because it hoisted the current DP row and distance row out of the inner loop and sized arrays by n-1 cities; the larger model's answer was shorter and terser. On a well-specified, textbook algorithm the cheaper model was fully sufficient. This single-function task says nothing about longer multi-step work such as debugging experiments, which was not compared. Limitations: one task, one attempt each, same vendor.
