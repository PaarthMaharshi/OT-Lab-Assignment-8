# Prompt log entry - Track I

- **AI tool + model version:** Claude Code, Claude Sonnet 5.5 (`claude-sonnet-5-5`), 2026-11-07
- **Prompt:** "just do everything else without asking me" (Track I from the assignment PDF)
- **Output:** `qalgs.py`, `qml.py`, `experiments.py`, 9 tests. Commit: [5b5fcdc](https://github.com/PaarthMaharshi/OT-Lab-Assignment-8/commit/5b5fcdc).
- **Verification:** Grover vs closed form; QFT vs DFT matrix; QPE bounds; DJ/BV deterministic outcomes; variational-classifier gradient vs finite differences; kernel PSD.

## Issues found (feed docs/failure_log.md)
1. **Inflated Shor success.** The first post-processing tried multiples of the candidate period, so a useless measurement (m = 0) still "found" the order by classical trial, giving 200/200. Detected because the rate was suspiciously perfect; fixed with strict decoding (about 50% per attempt, the expected behaviour).
2. **Wrong annealing gap, twice.** First reported about 1e-13 (the gap to the degenerate partner of the Max-Cut ground state, caused by global bit-flip symmetry); then a degenerate-skipping fix still failed its own test because of tiny tunnelling splittings. Correct quantity: gap between the two lowest levels of the bit-flip-even sector (0.43), which is where the evolution lives.
3. **Weak baseline.** The 8-hidden-unit MLP scored 0.64 +- 0.16 on circles, which would have made the quantum models look better than a fair classical baseline; added a (16,16) MLP (0.99).
4. **No-op leftover** line in `shor_order_finding` and a `20 if False else 10` in a test removed on review.
5. **Result kept despite being negative:** both quantum classifiers lose to the best classical model.
