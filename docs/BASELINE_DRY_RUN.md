# Verified Kaggle dry run

This is a pipeline check on the **first 10 problems** of each benchmark, not a full baseline
or a representative estimate of model quality. Full results still require all 164 HumanEval
and 500 MBPP-test problems per model. No training has been measured yet.

The source archive was downloaded as `coderl_results_base.zip`, but its four summaries all
have `limit: 10`. They are preserved unchanged under `results/dry_run/`. Model completions and
logs stay in the local source archive, outside Git.

## Scores (%)

| Model | Benchmark | Problems | Greedy pass@1 | Sampled pass@1 | Sampled pass@10 |
|---|---|---:|---:|---:|---:|
| Qwen2.5-Coder-0.5B-Instruct | HumanEval | 10 | 100.0 | 72.5 | 98.94 |
| Qwen2.5-Coder-0.5B-Instruct | MBPP-test | 10 | 30.0 | 36.5 | 49.57 |
| Qwen2.5-Coder-1.5B-Instruct | HumanEval | 10 | 80.0 | 78.0 | 98.93 |
| Qwen2.5-Coder-1.5B-Instruct | MBPP-test | 10 | 40.0 | 43.0 | 59.51 |

Each row contains one greedy completion plus 20 sampled completions per problem: **210 per
row, 840 total**. Sampled decoding uses temperature 0.8, top-p 0.95, and a 512-token budget.
Pass@10 uses the unbiased estimator. Independently regrouping all archived samples by task
and pass and recomputing pass@k exactly reproduces each summary (within floating-point error).
Sample counts, unique sample indices and recorded status totals also agree.

## Runtime and diagnostics

- Code commit: `937a5fb1b7cc6fa4f56567a4ac0bb7f73600a7d8`; clean working tree at evaluation.
- Actual generator: **HF**, CUDA float16, Tesla T4; torch 2.10.0, transformers 4.57.6,
  Python 3.13.16. vLLM 0.18.1 was installed but failed during engine startup.
- vLLM root error in the log: FlashInfer JIT compilation reached the linker, which reported
  `/usr/bin/ld: cannot find -lcuda: No such file or directory`. The HF fallback worked.
- Sandbox: subprocess, `eval` profile, isolation **none**. This does not provide OS-enforced
  filesystem/network isolation; it is not validation of safe RL-policy execution. The logs
  record no recognized secrets in the evaluation environment. See [SECURITY.md](../SECURITY.md).
- All 840 completions have `extraction_status: code_block`; no sandbox errors, timeouts,
  memory errors or integrity flags were recorded. This validates the recorded pipeline
  behavior; it does not prove the absence of every possible sandbox bypass.
- Across 800 sampled completions: 4 syntax errors (0.5%) and 1 runtime error (0.125%).
  All five occurred in the 1.5B model's outputs. One completion hit the token limit.
- Evaluation wall time: 155.23 seconds for 0.5B and 127.96 seconds for 1.5B, summing each
  model's two benchmark summaries. Downloads, installation and engine startup are excluded.

The 0.5B model's 100% greedy HumanEval score covers only ten selected tasks. Do not compare
it with published full-benchmark numbers or use it as evidence that 0.5B outperforms 1.5B.

## Next run

Follow [KAGGLE.md](KAGGLE.md). In cell 2, use `LIMIT = None`,
`GENERATION_BACKEND = "hf"`, `N_SAMPLES = 20`, and `DTYPE = "float16"`.
Download the full archive and verify four complete summaries before updating README baseline
scores or moving on to the rejection-sampling SFT and GRPO experiments.
