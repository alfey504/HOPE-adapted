# HOPE: Higher-Order Pruning of Experts

A second-order expert pruning method for Mixture-of-Experts (MoE) language models, which accounts for pairwise expert interactions.

HOPE formulates expert selection as a binary quadratic program over an interaction matrix **F**, where each entry captures the joint contribution of an expert pair weighted by their router gate values and output norms. This goes beyond first-order methods that score experts independently.

## Installation

**Prerequisites:** PyTorch must be installed separately to match your CUDA driver. See [pytorch.org](https://pytorch.org/get-started/locally/) for instructions. For example:

```bash
# Example for CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Then install HOPE:

```bash
git clone https://github.com/awslabs/HOPE.git
cd HOPE
pip install -e .
```

## Quick Start

HOPE uses a three-step pipeline: Calibrate, Solve, Prune

### 0. (Optional) Download and convert a model

`hope calibrate` and `hope prune` take a local model directory. To download a model from the HuggingFace Hub and save it in a different dtype:

```bash
python scripts/download_model.py \
    --model-id allenai/OLMoE-1B-7B-0924 \
    --out-path /path/to/olmoe-fp16 \
    --dtype float16 \
    --sanity-check
```

Use `--dtype float16` on GPUs without native bfloat16 support (e.g. T4, P100, V100), where it is considerably faster. Models trained in bfloat16 can overflow in float16; `--sanity-check` checks a short prompt, and `hope calibrate` raises an error if it collects non-finite statistics. If either happens, use `--dtype bfloat16` instead. The conversion needs roughly the model size in RAM, and twice the model size in disk space (HuggingFace cache + output).

### 1. Calibrate: Collect the F-matrix

Run calibration prompts through your model to collect expert interaction statistics:

```bash
hope calibrate \
    --model-path /path/to/moe-model \
    --prompts prompts.txt \
    --out-path observations.h5
```

Where `prompts.txt` has one prompt (string) per line. Alternatively, use `--prompts` to specify a `.json` file which contains a list of strings, or a list of lists of ints (pre-tokenized token IDs).

This produces an HDF5 file which contains the statistics needed to construct the F-matrix (expert usage and co-usage statistics).

### 2. Solve: Find the optimal pruning set

Solve the HOPE quadratic program to determine which experts to prune. The number of experts to prune can be specified as either a fraction (`--prune-frac`) or an integer count (`--prune-num`):

```bash
# Prune 25% of experts per layer:
hope solve \
    --obs-path observations.h5 \
    --prune-frac 0.25 \
    --out-path pruneset.json

# Or prune exactly 64 experts per layer:
hope solve \
    --obs-path observations.h5 \
    --prune-num 64 \
    --out-path pruneset.json
```

This materializes the F-matrix and solves the corresponding quadratic program to produce the actual prune-set (i.e. which experts to prune in each layer) as a JSON file. This is the main result of the HOPE algorithm.

### 3. Prune: Create the smaller model

Apply the pruning set to produce a smaller checkpoint:

```bash
hope prune \
    --model-path /path/to/moe-model \
    --pruneset-path pruneset.json \
    --out-path /path/to/pruned-model
```

The pruned model is a standard HuggingFace checkpoint and can be loaded directly:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("/path/to/pruned-model")
```

You may also prune your model with your own code instead of the script here (the main HOPE result is the prune-set JSON).

### Evaluating on your own prompts

Compare one or more pruned models against the original on a prompts file (same formats as `--prompts` above; no reference answers needed):

```bash
python scripts/eval_prompts.py \
    --original /path/to/moe-model \
    --pruned /path/to/pruned-model \
    --prompts eval_prompts.txt \
    --out-dir /path/to/eval
```

`--pruned` can be given several times (e.g. to compare HOPE against a baseline). Models are loaded one at a time. For each model it reports:
- **perplexity** on the prompt tokens (lower is better)
- **KL**: mean per-token KL divergence from the original's next-token distribution (lower is better)
- **top1**: fraction of positions where the most likely next token matches the original's (higher is better)
- **exact**: fraction of prompts where the greedy generation matches the original's exactly

Results are written to `summary.json`, and per-prompt generations and metrics to `generations.jsonl`.

The original model's results are cached in `<out-dir>/original_cache` (or `--original-cache-dir`) and reused on later runs, so evaluating new pruned models doesn't rerun the original. Only new prompts are computed, and the cache is cleared automatically if the original model changes; pass `--recompute-original` to force a fresh run.

### First-order baselines

HOPE's calibration data also supports several first-order baselines (REAP, EAN, MAN, freq):

```bash
hope baselines \
    --obs-path observations.h5 \
    --prune-frac 0.25 \
    --method reap \
    --out-path pruneset_reap.json
```

## Python API

```python
from hope.calibrate import calibrate
from hope.solve import solve
from hope.prune import prune_model

# Step 1
calibrate("path/to/model", ["prompt 1", "prompt 2", ...], "obs.h5")

# Step 2: prune by fraction or by count
solve("obs.h5", "pruneset.json", prune_frac=0.25)
# OR
solve("obs.h5", "pruneset.json", prune_num=64)

# Step 3
prune_model("path/to/model", "pruneset.json", "path/to/pruned")
```

See `examples/quickstart.py` for a complete end-to-end example.

## Supported Models

HOPE's calibration step hooks into a HuggingFace MoE's expert Module. The observer requires that models use the stacked-experts layout, where all expert weights (in each MoE layer) are stored as a 3D tensor (`gate_up_proj` with shape `(num_experts, 2 * intermediate_size, hidden_size)` and `down_proj` with shape `(num_experts, hidden_size, intermediate_size)` on a single module). HOPE's code is tested on `transformers==5.2.0` and on the following model architectures:

- Qwen3 MoE (e.g. Qwen3-30B-A3B)
- Qwen3-Next
- Qwen3.5 MoE (e.g. Qwen3.5-35B-A3B, Qwen3.5-122B-A10B)
- GLM-4.5-Air
- OLMoE (e.g. OLMoE-1B-7B-0924)

## Citation

```bibtex
@article{tseng2026hope,
    title={Higher-Order Pruning of Experts in Mixture-of-Experts Language Models},
    author={Tseng, Alex M. and Kaul, Prannay and Zancato, Luca and Xia, Wei and Soatto, Stefano},
    journal={arXiv preprint arXiv:2609.18916},
    year={2026},
    url={https://arxiv.org/abs/2609.18916},
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).
