# Running the base-model evaluation on Kaggle

Notebook: [`notebooks/kaggle_eval_base.ipynb`](../notebooks/kaggle_eval_base.ipynb). It evaluates
Qwen2.5-Coder-0.5B-Instruct and -1.5B-Instruct (no training) on HumanEval and MBPP-test and
produces `coderl_results_base.zip`.

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
4. **Run All.** The cells in order:
   1. `nvidia-smi` and the Python version.
   2. Configuration: models, `LIMIT`, `N_SAMPLES`, `DTYPE=float16`, GPU memory fraction.
   3. Clone the repo and run `pip install -e ".[gen]"`. This pins `vllm==0.18.1` and replaces
      Kaggle's torch with torch 2.10. vLLM is then test-imported in a fresh interpreter. If the
      install or the import fails, the run uses the Hugging Face backend and prints a warning.
   4. `apt-get install bubblewrap` (failure here is not fatal), then print the sandbox
      isolation level.
   5. Load the secrets.
   6. Build the datasets and download both models. This cell runs in the kernel, which holds
      `HF_TOKEN`.
   7. Evaluate. Each model runs in its own process on its own T4. Progress is printed every
      minute, and the full logs are in `results/base/logs/<tag>.log`. These processes get **no
      secrets** in their environment (see *Security* below).
   8. Print the results table (also written to `results/base/RESULTS.md`) and upload to W&B.
   9. Zip the results to `/kaggle/working/coderl_results_base.zip`.
5. **Download** the zip from the *Output* tab. Unzip it in the repo root and commit only the
   summaries: `git add results/base/*/*/summary.json`. `samples.jsonl.gz` and the logs are
   gitignored.

Tip: set `LIMIT = 10` in cell 2 for a dry run of about 5 minutes before the full run.

Expected time with vLLM: about 10 minutes to install, then roughly 30–90 minutes of evaluation.
Each model runs 664 problems × 21 completions. With the HF fallback, expect several hours.

## Why vLLM 0.18.1

- Python 3.10–3.13 wheels (`cp38-abi3`). Kaggle's Python is somewhere in that range. The
  newest vLLM releases need Python ≥ 3.11.
- torch 2.10 / CUDA 12.8 wheels still include kernels for the T4 (compute capability 7.5).
- Reported working on Kaggle's 2× T4 (third-party reports). On SM75, vLLM cannot use
  FlashAttention-2 and picks a Triton attention backend by itself. The T4 has no bfloat16, so
  the notebook passes `--dtype float16`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` / `No available memory for the cache blocks` | In cell 2, lower `GPU_MEMORY_UTILIZATION` (0.7, then 0.6) and/or `MAX_MODEL_LEN` (1536). Prompts plus 512 new tokens fit in 1536. |
| `Bfloat16 is only supported on GPUs with compute capability of at least 8.0` | `DTYPE` must be `"float16"` (the default in cell 2). |
| vLLM install or import fails (cell 3 prints the WARNING block) | The run continues with the HF backend. To retry vLLM, *Run → Restart & clear cell outputs* and run again. If it still fails, set `DTYPE="float16"` and keep the HF fallback. The results are comparable; HF is only slower. |
| CUDA graph capture errors or hangs at start-up | Add `"--enforce-eager"` to the command list in `eval_cmd` in cell 7. |
| Only one GPU visible | The models run one after the other on GPU 0. Check that *Accelerator* is **T4 x2**. |
| `pip` dependency-conflict warnings for preinstalled Kaggle packages | Harmless. The evaluation uses only CodeRL's own dependencies. |
| Dataset download fails | Internet must be **On**. Add `HF_TOKEN` if the Hub rate-limits you. |
| A model's process fails | Cell 7 prints the last 40 log lines. The full log is in `results/base/logs/<tag>.log`. |
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
