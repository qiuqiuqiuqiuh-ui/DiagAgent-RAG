# %% [markdown]
# # DiagAgent-RAG GPU benchmark
#
# Select a GPU runtime and put the repository at `MyDrive/DiagAgent-RAG`.
# Add the OpenAI-compatible key as the Colab secret `DIAGAGENT_LLM_API_KEY`
# and grant this notebook access. The benchmark runs all 54 labeled cases through
# four retrieval modes and calls the configured live LLM. Enable `use_judge`
# for additional faithfulness calls.
#
# The notebook creates an isolated Python 3.11 environment under `/content`.
# The corpus, portable index and evaluation reports live on Drive.

# %%
import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata

drive.mount("/content/drive")
repo = Path("/content/drive/MyDrive/DiagAgent-RAG")
profile = "configs/colab.yaml"
dataset = "data/eval/golden_bench.jsonl"
use_judge = False
if not (repo / profile).is_file() or not (repo / dataset).is_file():
    raise FileNotFoundError(f"Place the repository with {profile} and {dataset} at {repo}")
subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], check=True)

# %%
api_key = userdata.get("DIAGAGENT_LLM_API_KEY")
if not api_key or not api_key.strip():
    raise RuntimeError("Set the DIAGAGENT_LLM_API_KEY Colab secret before the live run")
run_env = {
    key: value
    for key, value in os.environ.items()
    if not key.upper().startswith(("DIAGAGENT_", "LANGSMITH_", "LANGCHAIN_"))
    and key != "PYTHONPATH"
}
run_env["DIAGAGENT_LLM__API_KEY"] = api_key
del api_key
run_env["DIAGAGENT_LOG_LEVEL"] = "WARNING"

# %%
venv = Path("/content/diagagent-rag-venv")
python = venv / "bin" / "python"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
uv = [sys.executable, "-m", "uv"]
if not python.is_file():
    subprocess.run([*uv, "venv", "--python", "3.11", str(venv)], check=True)
subprocess.run(
    [*uv, "pip", "install", "--python", str(python), "--torch-backend=auto", ".[huggingface]"],
    cwd=repo,
    check=True,
)
cuda_check = (
    "import torch; "
    "assert torch.cuda.is_available(), 'A CUDA-enabled Torch runtime is required'; "
    "print('CUDA device:', torch.cuda.get_device_name(0))"
)
subprocess.run([str(python), "-c", cuda_check], env=run_env, check=True)

# %%
subprocess.run(
    [str(python), "scripts/ingest.py", "--config", profile],
    cwd=repo,
    env=run_env,
    check=True,
)

# %%
command = [str(python), "scripts/evaluate.py", "--config", profile, "--dataset", dataset]
if use_judge:
    command.append("--judge")
subprocess.run(command, cwd=repo, env=run_env, check=True)
