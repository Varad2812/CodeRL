# Running the base-model evaluation on Kaggle

Notebook: [`notebooks/kaggle_eval_base.ipynb`](../notebooks/kaggle_eval_base.ipynb). It evaluates
Qwen2.5-Coder-0.5B-Instruct and -1.5B-Instruct (no training) on HumanEval and MBPP-test and
produces `coderl_results_base.zip` for a full run or `coderl_results_dry_run.zip` for a limited run.

## Steps

1. **Import the notebook.** On kaggle.com: *Create → New Notebook*, then *File → Import
   Notebook* and upload `notebooks/kaggle_eval_base.ipynb`, or paste its GitHub URL.
2. **Settings** (right-hand panel):
   - *Accelerator* → **GPU T4 x2**
   - *Internet* → **On**. This needs a phone-verified account (*Settings → Phone verification*).
3. **Secrets** (*Add-ons → Secrets*). Both are optional. Add each one and tick *Attach to
   notebook*:
   - `WANDB_API_KEY`: logs the results to W&B, project `coderl`, job type `eval`.
   - `HF_TOKEN`: authenticated Hub downloads, which avoids rate limits.
4. **Choose the run in cell 2.** The default is `LIMIT = 10` and
   `GENERATION_BACKEND = "hf"`. This uses the backend that completed the saved T4 dry run,
   skipping the failing vLLM installation/startup path. `"auto"` attempts vLLM with HF fallback.
   For the full baseline, set **`LIMIT = None`**, keep **`N_SAMPLES = 20`**, and keep
   **`DTYPE = "float16"`**. Then **Run All**. The cells in order:
   1. `nvidia-smi` and the Python version.
   2. Configuration: models, `LIMIT`, `N_SAMPLES`, `DTYPE=float16`, GPU memory fraction.
   3. Clone the repo and install `.[hf]` for explicit HF mode. In `"auto"` mode, install
      `.[gen]` (vLLM 0.18.1 / torch 2.10) and probe vLLM in a fresh interpreter; installation,
      import or engine startup failures fall back to HF with a warning.
   4. `apt-get install bubblewrap` (failure here is not fatal), then print the sandbox
      isolation level.
   5. Load the secrets.
   6. Build the datasets and download both models. This cell runs in the kernel, which holds
      `HF_TOKEN`.
   7. Evaluate. Each model runs in its own process on its own T4. Progress is printed every
      minute, and the full logs are in `<RESULTS_DIR>/logs/<tag>.log`. These processes get **no
      secrets** in their environment (see *Security* below).
   8. Print the results table (also written to `<RESULTS_DIR>/RESULTS.md`) and upload to W&B.
   9. Zip the selected results folder to `/kaggle/working/coderl_results_base.zip` (full run)
      or `/kaggle/working/coderl_results_dry_run.zip` (limited run).
5. **Download** the zip from the *Output* tab. A full run must have **four summaries**:
   two models times two benchmarks. Check `num_problems` is **164** for HumanEval and **500**
   for MBPP-test, `limit` is **null**, sampled `n` is **20**, and `sandbox_error` counts are zero.
   Only then import the full summaries into `results/base/` and publish baseline numbers.
   `samples.jsonl.gz` and logs are gitignored; keep the zip locally for audit.

Dry runs write to **`results/dry_run/`** and cannot overwrite `results/base/`. The verified
10-problem run is documented in [BASELINE_DRY_RUN.md](BASELINE_DRY_RUN.md). Its scores are
smoke-test diagnostics; the main README results remain TBD until the complete benchmarks run.

Tip: the default `LIMIT = 10` runs a small end-to-end check before spending GPU time on the full baseline.

Expected time with vLLM: about 10 minutes to install, then roughly 30–90 minutes of evaluation.
Each model runs 664 problems × 21 completions. With the HF fallback, expect several hours.

## Why vLLM 0.18.1

- Python 3.10–3.13 wheels (`cp38-abi3`). Kaggle's Python is somewhere in that range. The
  newest vLLM releases need Python ≥ 3.11.
- torch 2.10 / CUDA 12.8 wheels still include kernels for the T4 (compute capability 7.5).
- The saved T4 dry run installed this version, but engine startup failed in FlashInfer's
  JIT linker: `/usr/bin/ld: cannot find -lcuda`. An import probe alone does not prove the
  engine works. HF fallback completed all four evaluations; HF is the notebook default.
- T4 evaluation uses `--dtype float16`. Auto-selected attention backends depend on vLLM and
  the runtime; do not assume Triton will be selected.

## Troubleshooting

| Symptom | Fix |
|---|---|
| HF: `CUDA out of memory` | In cell 2 lower `HF_BATCH_SIZE` from 80 to 40 or 20. vLLM memory settings do not affect HF. |
| vLLM: `CUDA out of memory` / `No available memory for the cache blocks` | In cell 2, lower `GPU_MEMORY_UTILIZATION` (0.7, then 0.6) and/or `MAX_MODEL_LEN` (1536). Prompts plus 512 new tokens fit in 1536. |
| `Bfloat16 is only supported on GPUs with compute capability of at least 8.0` | `DTYPE` must be `"float16"` (the default in cell 2). |
| vLLM install, import or engine startup fails | The run continues with the HF backend. To retry vLLM, *Run → Restart & clear cell outputs* and run again. If it still fails, set `DTYPE="float16"` and keep the HF fallback. The results are comparable; HF is only slower. |
| `Ninja build failed` / `cannot find -lcuda` | Set `GENERATION_BACKEND="hf"` in cell 2 and rerun from cell 2. This bypasses vLLM/FlashInfer; no driver or dependency changes are needed for that route. |
| CUDA graph capture errors or hangs at start-up | Add `"--enforce-eager"` to the command list in `eval_cmd` in cell 7. |
| Only one GPU visible | The models run one after the other on GPU 0. Check that *Accelerator* is **T4 x2**. |
| `pip` dependency-conflict warnings | Check which packages conflict. If torch, transformers, vLLM or CUDA packages are affected, do not assume the warning is harmless. Explicit HF mode avoids the vLLM stack replacement. |
| Dataset download fails | Internet must be **On**. Add `HF_TOKEN` if the Hub rate-limits you. |
| A model's process fails | Cell 7 prints the last 40 log lines. The full log is in `<RESULTS_DIR>/logs/<tag>.log`. |
| Session timeout (12 h) | Run one model per session. Delete the other entry from `MODELS` in cell 2. |

## Security

Generated code runs in the subprocess sandbox. Kaggle notebooks run inside a container that
usually blocks the user namespaces bubblewrap needs, so isolation often drops to `unshare` or
`none`. The level reached is printed in cell 4 and stored in every `summary.json` under
`sandbox.isolation`.

At `none`, generated code still has rlimits, the in-process audit hook and a scrubbed
environment. It can still read files and use the network as the notebook user, including the
parent process's environment through `/proc`. For that reason the evaluation processes start
without `WANDB_API_KEY` / `HF_TOKEN`. The models are downloaded beforehand, and W&B logging
runs afterwards from the summaries (`python -m coderl.eval.report results/base --wandb`). For
untrusted (RL-trained) policies, prefer a machine where `python -m coderl.sandbox info` reports
`bwrap`. See [SECURITY.md](../SECURITY.md).
