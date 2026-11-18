"""Generate notebooks/kaggle_eval_base.ipynb (edit here, then run: uv run python scripts/make_kaggle_notebook.py).

Cell code is plain Python (no IPython magics) so tests/test_notebook.py can compile it.
"""

import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

md = new_markdown_cell
code = new_code_cell

cells = [
    md("""# CodeRL — base-model evaluation on Kaggle (Phase 2a)

Measures **Qwen2.5-Coder-0.5B-Instruct** and **Qwen2.5-Coder-1.5B-Instruct** (no training) on
**HumanEval** (164) and **MBPP-test** (500): greedy pass@1, and unbiased pass@1 / pass@10 from
n=20 samples at T=0.8, top_p=0.95. Every completion runs in the CodeRL sandbox (`eval` limit
profile).

### Kaggle settings (right-hand panel → *Settings*) — set these BEFORE running
| Setting | Value |
|---|---|
| **Accelerator** | **GPU T4 x2** |
| **Internet** | **On** (needs a phone-verified Kaggle account) |
| **Persistence** | not needed |
| **Secrets** (*Add-ons → Secrets*) | `WANDB_API_KEY` (optional: logs to W&B project `coderl`), `HF_TOKEN` (optional: avoids Hub rate limits). Tick *Attach to notebook* for each. |

Then **Run All**. Expected time: ~10 min install, then roughly 30–90 min of evaluation
(both models run in parallel, one per T4). If vLLM cannot be installed, it falls back to Hugging
Face `transformers` automatically (much slower: several hours). Troubleshooting:
[`docs/KAGGLE.md`](https://github.com/Varad2812/CodeRL/blob/main/docs/KAGGLE.md).

**Output:** `/kaggle/working/coderl_results_base.zip` (*Output* tab → download). Unzip it into
the repo root; only the `summary.json` files are committed (samples and logs are gitignored).
"""),
    md("## 1. GPU and Python"),
    code("""import subprocess
import sys

print("Python", sys.version)
smi = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
print(smi.stdout or smi.stderr)
if smi.returncode != 0:
    print("WARNING: no GPU visible. Settings -> Accelerator -> 'GPU T4 x2', then restart.")"""),
    md("## 2. Configuration (edit here if needed)"),
    code("""REPO_URL = "https://github.com/Varad2812/CodeRL"
REPO_REF = "main"                      # branch, tag or commit to evaluate
WORKDIR = "/kaggle/working/CodeRL"
MODELS = {                             # results/base/<tag>/
    "qwen0.5b": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
}
BENCHMARKS = "humaneval,mbpp_test"
LIMIT = None                           # e.g. 10 for a quick dry run of the whole notebook
N_SAMPLES = 20
DTYPE = "float16"                      # T4 has no bfloat16
GPU_MEMORY_UTILIZATION = 0.85          # lower (0.7) if vLLM runs out of memory
MAX_MODEL_LEN = 2048"""),
    md("""## 3. Clone the repo and install
Installs CodeRL with the `gen` extra (pins `vllm==0.18.1`, which replaces Kaggle's torch with
torch 2.10). vLLM is then imported in a *fresh* interpreter; if that fails, the run uses the
Hugging Face backend instead (printed loudly below)."""),
    code("""import os
import subprocess
import sys


def sh(cmd, check=True):
    print("$", cmd, flush=True)
    rc = subprocess.run(cmd, shell=True).returncode
    if check and rc != 0:
        raise RuntimeError(f"command failed ({rc}): {cmd}")
    return rc


if not os.path.isdir(os.path.join(WORKDIR, ".git")):
    sh(f"git clone --depth 1 --branch {REPO_REF} {REPO_URL} {WORKDIR}")
else:
    sh(f"git -C {WORKDIR} fetch --depth 1 origin {REPO_REF} && git -C {WORKDIR} checkout -q FETCH_HEAD")
os.chdir(WORKDIR)
sh("git log --oneline -1")

PY = sys.executable
if sh(f'{PY} -m pip install -q -e ".[gen]"', check=False) != 0:
    print("\\n*** WARNING: installing the [gen] extra (vLLM) failed; installing [hf] only ***\\n")
    sh(f'{PY} -m pip install -q -e ".[hf]"')

probe = subprocess.run(
    [PY, "-c", "import vllm, torch; print(vllm.__version__, torch.__version__, torch.version.cuda, torch.cuda.is_available())"],
    capture_output=True, text=True,
)
if probe.returncode == 0 and probe.stdout.strip().endswith("True"):
    BACKEND = "vllm"
    print("vLLM OK (vllm, torch, cuda, gpu):", probe.stdout.strip())
else:
    BACKEND = "hf"
    print("\\n" + "*" * 80)
    print("WARNING: vLLM is not usable here -> falling back to the Hugging Face backend.")
    print("Results are equivalent but generation is MUCH slower (hours). Details:")
    print((probe.stdout + probe.stderr)[-3000:])
    print("*" * 80 + "\\n")
print("generation backend:", BACKEND)"""),
    md("""## 4. Sandbox isolation
Tries to install `bubblewrap` (non-fatal). The isolation level actually used is also stored
in every `summary.json` (`sandbox.isolation`: `bwrap` > `unshare` > `none`)."""),
    code("""import json

if sh("apt-get install -y -qq bubblewrap > /dev/null 2>&1", check=False) != 0:
    sh("(apt-get update -qq && apt-get install -y -qq bubblewrap) > /dev/null 2>&1", check=False)

info = subprocess.run([PY, "-m", "coderl.sandbox", "info"], capture_output=True, text=True)
print(info.stdout or info.stderr)
try:
    ISOLATION = json.loads(info.stdout[info.stdout.index("{"):])["subprocess_isolation"]
except (ValueError, KeyError):
    ISOLATION = "unknown"
print("sandbox isolation:", ISOLATION)
if ISOLATION != "bwrap":
    print("NOTE: weaker sandbox isolation than bwrap (common inside Kaggle's container). OK for "
          "evaluating base models: the eval processes below get no secrets, and the level is "
          "recorded in every summary.json.")"""),
    md("## 5. Secrets (optional)"),
    code("""try:
    from kaggle_secrets import UserSecretsClient

    _secrets = UserSecretsClient()
    for _name in ("WANDB_API_KEY", "HF_TOKEN"):
        try:
            os.environ[_name] = _secrets.get_secret(_name)
            print(f"{_name}: loaded")
        except Exception:
            print(f"{_name}: not set (optional)")
except Exception as exc:
    print("Kaggle secrets unavailable (not on Kaggle?):", repr(exc))"""),
    md("## 6. Build the datasets and pre-download the models (HF Hub)"),
    code("""sh(f"{PY} -m coderl.data.build --out data/processed")
sh("wc -l data/processed/*.jsonl")
# Download weights here, in the notebook kernel (which holds HF_TOKEN), so the evaluation
# processes below can run WITHOUT any secrets in their environment.
for _model in MODELS.values():
    sh(f"{PY} -c \\"from huggingface_hub import snapshot_download as s; print(s('{_model}'))\\"")"""),
    md("""## 7. Evaluate both base models
With two GPUs, each model runs in its own process on its own T4 (`CUDA_VISIBLE_DEVICES`);
otherwise they run one after the other. Progress (JSON lines) is printed every minute; full
logs are in `results/base/logs/`.

The evaluation processes get **no secrets** (`WANDB_API_KEY` / `HF_TOKEN` are removed from their
environment): unless the sandbox reached `bwrap` isolation, generated code could read its
parent's environment. W&B logging happens afterwards, in step 8."""),
    code("""import time

gpus = [l for l in subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout.splitlines() if l.startswith("GPU")]
N_GPUS = max(1, len(gpus))
SECRETS = ("WANDB_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "KAGGLE_KEY")
os.makedirs("results/base/logs", exist_ok=True)


def eval_cmd(tag, model):
    cmd = [
        PY, "-m", "coderl.eval.run_benchmarks",
        "--model", model, "--backend", BACKEND, "--benchmarks", BENCHMARKS,
        "--out", f"results/base/{tag}", "--dtype", DTYPE, "--n-samples", str(N_SAMPLES),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--batch-size", "64", "--hf-batch-size", "80",
        "--sandbox-workers", "2" if N_GPUS > 1 else "4",
    ]
    return cmd + (["--limit", str(LIMIT)] if LIMIT else [])


def start(tag, model, gpu):
    env = {k: v for k, v in os.environ.items() if k not in SECRETS}
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
    log = open(f"results/base/logs/{tag}.log", "w")
    print(f"[{tag}] GPU {gpu}: {' '.join(eval_cmd(tag, model))}", flush=True)
    return subprocess.Popen(eval_cmd(tag, model), env=env, stdout=log, stderr=subprocess.STDOUT)


def last_progress(tag):
    try:
        lines = open(f"results/base/logs/{tag}.log").read().splitlines()
    except OSError:
        return ""
    keep = [l for l in lines if '"eval_' in l or "WARNING" in l or "generator_fallback" in l]
    return (keep or lines or [""])[-1][:400]


def wait(procs):
    while any(p.poll() is None for p in procs.values()):
        time.sleep(60)
        for tag in procs:
            print(f"[{tag}] {last_progress(tag)}", flush=True)
    failed = [t for t, p in procs.items() if p.returncode != 0]
    for tag in failed:
        print(f"\\n===== {tag} FAILED (exit {procs[tag].returncode}); last log lines =====")
        print("\\n".join(open(f"results/base/logs/{tag}.log").read().splitlines()[-40:]))
    if failed:
        raise RuntimeError(f"evaluation failed for {failed}; see docs/KAGGLE.md troubleshooting")


t0 = time.time()
items = list(MODELS.items())
if N_GPUS >= 2:
    wait({tag: start(tag, model, i) for i, (tag, model) in enumerate(items)})
else:
    for tag, model in items:
        wait({tag: start(tag, model, 0)})
print(f"all evaluations done in {(time.time() - t0) / 60:.1f} min")"""),
    md("## 8. Results table (+ W&B upload if `WANDB_API_KEY` was loaded)"),
    code(
        """sh(f"{PY} -m coderl.eval.report results/base --out results/base/RESULTS.md --wandb")"""
    ),
    md("## 9. Package the results for download"),
    code("""import shutil

archive = shutil.make_archive("/kaggle/working/coderl_results_base", "zip", root_dir=WORKDIR, base_dir="results/base")
print("download from the Output tab:", archive, f"({os.path.getsize(archive) / 1e6:.1f} MB)")
print("then, in your local clone:  unzip coderl_results_base.zip  &&  git add results/base/*/*/summary.json")"""),
]

nb = new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
}
nbformat.validate(nb)
OUT = sys.argv[1] if len(sys.argv) > 1 else "notebooks/kaggle_eval_base.ipynb"
nbformat.write(nb, OUT)
sys.stdout.write(f"wrote {OUT}\n")
