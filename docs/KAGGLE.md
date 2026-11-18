# Run the base-model evaluation on Kaggle

Notebook: [kaggle_eval_base.ipynb](../notebooks/kaggle_eval_base.ipynb).
Evaluate Qwen2.5-Coder-0.5B-Instruct first, download its results, then evaluate the 1.5B model.
The full baseline requires 164 HumanEval and 500 MBPP-test problems for each model.

## Prepare a run

1. Import the updated notebook using File -> Import Notebook.
2. Under Session options choose GPU T4 x2, Internet On, and **Persistence -> Files only**.
   Persistence helps carry working files between interactive sessions; keep downloaded backups.
3. Optional: attach HF_TOKEN and WANDB_API_KEY under Add-ons -> Secrets. Do not put keys
   into code cells. Model downloads run before evaluation; the evaluation process receives
   neither key, and W&B logging happens afterwards.
4. In cell 2 choose:

   ```python
   MODEL_TAG = "qwen0.5b"  # use "qwen1.5b" only after downloading the 0.5B results
   LIMIT = None  # 10 for a small dry run; None for the full baseline
   GENERATION_BACKEND = "hf"
   HF_BATCH_SIZE = 20
   EVAL_BATCH_SIZE = 8
   ```

   Keep N_SAMPLES=20 and DTYPE="float16". Keep REPO_REF set to the recovery branch supplied
   by this notebook so the evaluator supports --resume. Changes to source, data, model,
   sampling or batch settings are detected before checkpoint reuse.
5. For a long unattended run, use **Save Version -> Save & Run All**. Kaggle starts a separate,
   clean session and saves the completed version with its outputs. This does not reuse files
   from an interactive session. To retry an interrupted interactive run using retained files,
   run the cells in the same interactive workspace with unchanged settings instead.
6. Check progress in section 7. HF logs generation progress inside each 8-problem batch;
   eval_batch_saved confirms its execution results were checkpointed. A status line can repeat
   while a generation call is running. Runtime varies with problem length; allow several hours.
7. Download the selected model's archive from Output after completion:
   `coderl_results_base_qwen0.5b.zip` or `coderl_results_base_qwen1.5b.zip`.
   Dry-run names start with `coderl_results_dry_run_` and must not be published as baseline scores.
8. For the second model, change MODEL_TAG and repeat the run and download. Validate two full
   summaries per model, four in total: num_problems=164/500, limit=null, sampled n=20, and no
   sandbox_error statuses. Keep the sample archives locally; only summary JSONs go into Git.

[Notebook versions and Save & Run All](https://www.kaggle.com/docs/notebooks) and
[Files only persistence (Kaggle staff)](https://www.kaggle.com/discussions/product-feedback/355440).

## Recovery and limits

Every completed generation/execution batch is atomically written as compressed JSON under
`results/<run>/<model>/<benchmark>/checkpoints/`. The notebook passes --resume automatically:
a matching retry loads completed batches instead of regenerating them. Final summary.json
and samples.jsonl.gz are published atomically only after the benchmark's enabled passes finish.

Checkpoint manifests fingerprint the model/backend, complete problem contents, decoding and
batch settings, sandbox limits, code commit and environment metadata. Mismatched protocols,
legacy summaries without manifests, and incomplete/duplicate saved samples are rejected.
Use a separate output directory for a deliberately different protocol. Do not combine samples
from retries that used different settings into a single pass@k estimate.

If a model process fails, section 7 still creates a `*_recovery.zip` containing the completed
batch files, logs and any finished summaries. Download it before ending the session. A failed
run's archive may be incomplete; inspect summaries before claiming a full baseline. Retained
checkpoints or an unpacked recovery archive can be reused by the same code/environment and
settings. Without the files, resume is impossible. Neither disk checkpoints nor Files only
persistence guarantee recovery from every runtime deletion or crash.

For resumed results, pass generation/execution times include saved batch times; wall_seconds
measures the current invocation. resumed_batches records reused work in each pass.

## Why HF is the default

The verified 10-problem T4 run completed via HF. vLLM 0.18.1 installed but FlashInfer engine
startup failed at the CUDA linker: `cannot find -lcuda`. Explicit HF mode skips that path;
`GENERATION_BACKEND="auto"` remains an opt-in vLLM attempt with HF fallback. This project
still pins vLLM 0.18.1 for that route. GPU validation of the smaller full-run batch remains
necessary; offline recovery tests cannot establish GPU memory requirements.

The original full 1.5B run exhausted a T4 with HF_BATCH_SIZE=80. Twenty sequences per forward
uses less peak memory and is the new default. EVAL_BATCH_SIZE=8 controls checkpoint cadence;
it is distinct from the number of sequences sent through the model at once. N_SAMPLES=20
remains unchanged so sampled pass@1 and pass@10 use the intended estimator.

## Troubleshooting

| Symptom | Action |
|---|---|
| HF CUDA out of memory | Preserve the recovery archive first. Confirm HF_BATCH_SIZE=20; changing memory/decoding settings requires a new output directory because the checkpoint protocol must match. Do not restart the full evaluation with 80. |
| No summaries after a session reset | Check retained checkpoint files or downloaded recovery archives. Old notebook output text alone cannot restore the JSON results. |
| Checkpoint protocol differs | Restore the exact original code/environment/settings or use a new output directory. Do not delete completed baseline files just to silence this check. |
| Cannot find -lcuda / vLLM Ninja build failed | Keep GENERATION_BACKEND="hf" to avoid the vLLM startup path. |
| pip dependency conflicts | Inspect affected packages. Gradio/Diffusers are not used here; conflicts affecting torch, transformers or CUDA need investigation. |
| Download fails | Verify Internet On. HF_TOKEN can help with Hub rate limits. |
| A process fails | Read the actual traceback above the final evaluation-failed wrapper and download the recovery archive. |

## Sandbox

Kaggle may block the namespaces required by bubblewrap and unshare. The subprocess backend
then reports isolation=none: resource limits and an in-process audit hook remain, but OS-enforced
filesystem/network isolation is absent. The achieved level is stored in summaries. Keep secrets
out of the evaluation process. RL-trained or otherwise adversarial policies need a host with
strong OS isolation; see [SECURITY.md](../SECURITY.md).
