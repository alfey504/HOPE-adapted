"""
Evaluates pruned models against the original model on a file of prompts.

The prompts need no reference answers: the original (unpruned) model serves as
the reference. For each model, reports:
    - perplexity on the prompt tokens (lower is better)
    - mean per-token KL(original || pruned) of next-token distributions (lower
      is better; 0 means identical predictions)
    - top-1 agreement: fraction of positions where the most likely next token
      matches the original's (higher is better)
    - exact-match rate: fraction of prompts where the greedy generation is
      identical to the original's

Models are loaded one at a time, so only one needs to fit on the GPUs.

The original model's results (generations, perplexities, and next-token
log-probabilities) are cached in `--original-cache-dir` and reused across runs,
so evaluating new pruned models doesn't rerun the original. Entries are keyed
by prompt tokens and `--max-new-tokens`, so changing `--limit` or adding
prompts only computes the missing ones. The cache is cleared automatically if
the original model's files change; use `--recompute-original` to force it. The
log-probabilities are stored as float16 (roughly 2 bytes x vocab size per
prompt token), so KL values below ~1e-3 (including slightly negative values)
are within rounding error of 0.

Outputs (in `--out-dir`):
    - `summary.json`: aggregate metrics per model
    - `generations.jsonl`: one line per prompt with each model's generation and
      per-prompt metrics

Usage:
    python scripts/eval_prompts.py --original /path/to/original \
        --pruned /path/to/pruned_model --prompts eval_prompts.txt \
        --out-dir /path/to/eval
"""

import gc
import glob
import hashlib
import json
import math
import os
import shutil
import click

# Reduce CUDA memory fragmentation when loading models back to back; must be
# set before torch initializes CUDA
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import tqdm
import transformers

from hope.calibrate import encode_prompt


def _load_prompts(path):
    """
    Loads prompts from a .json (list of strings) or .txt (one per line) file.
    """
    with open(path) as f:
        if path.endswith(".json"):
            return json.load(f)
        return [line.strip() for line in f if line.strip()]


def _load_model(path):
    """
    Loads a model across available devices.
    """
    model = transformers.AutoModelForCausalLM.from_pretrained(
        path, device_map="auto", torch_dtype="auto", trust_remote_code=True,
    )
    model.eval()
    return model


def _model_fingerprint(path):
    """
    Returns a fingerprint of a model directory (path, config, and weight file
    names/sizes/modification times), used to invalidate the original's cache.
    """
    with open(os.path.join(path, "config.json")) as f:
        config = f.read()
    weights = sorted(
        glob.glob(os.path.join(path, "*.safetensors"))
        + glob.glob(os.path.join(path, "*.bin"))
    )
    return {
        "path": os.path.abspath(path),
        "config": config,
        "weights": [
            [os.path.basename(w), os.path.getsize(w), os.path.getmtime(w)]
            for w in weights
        ],
    }


def _prompt_key(ids, max_new_tokens):
    """
    Returns the cache key for a tokenized prompt.
    """
    return hashlib.sha256(
        json.dumps([ids, max_new_tokens]).encode()
    ).hexdigest()


def _prepare_cache(cache_dir, original, recompute):
    """
    Creates the original model's cache directory, clearing it if forced or if
    the original model has changed since the cache was written.
    """
    meta_path = os.path.join(cache_dir, "meta.json")
    fingerprint = _model_fingerprint(original)
    if os.path.exists(cache_dir):
        if recompute:
            print("Clearing original model cache (--recompute-original)")
            shutil.rmtree(cache_dir)
        else:
            meta = None
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
            if meta != fingerprint:
                print(
                    "Original model changed since the cache was written; "
                    "clearing %s" % cache_dir
                )
                shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(fingerprint, f)


def _run_model(
    path, name, tokenizer, input_ids_list, keys, max_new_tokens, cache_dir,
    is_original,
):
    """
    Loads a model, evaluates it with `_eval_model`, and frees its memory
    before returning, so that only one model is ever on the GPUs.
    """
    print("Loading %s from %s ..." % (name, path))
    model = _load_model(path)
    try:
        return _eval_model(
            model, tokenizer, input_ids_list, keys, max_new_tokens, cache_dir,
            is_original, name,
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@torch.no_grad()
def _eval_model(
    model, tokenizer, input_ids_list, keys, max_new_tokens, cache_dir,
    is_original, name,
):
    """
    Runs a model over the given prompts. For the original model, writes each
    prompt's result and next-token log-probs to `cache_dir` (as
    `<key>.pt`); for pruned models, compares against those entries.
    Returns a list of per-prompt result dictionaries.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    results = []
    for ids, key in tqdm.tqdm(
        list(zip(input_ids_list, keys)), desc=name
    ):
        ids_ten = torch.tensor([ids], device=model.device)
        cache_path = os.path.join(cache_dir, key + ".pt")
        entry = {}
        res = {
            "nll_sum": None, "num_tokens": len(ids) - 1, "kl_sum": None,
            "agree_sum": None, "nonfinite": False,
        }

        if len(ids) >= 2:
            logits = model(input_ids=ids_ten).logits[0, :-1]
            if not torch.isfinite(logits).all():
                res["nonfinite"] = True
            else:
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                targets = ids_ten[0, 1:].to(log_probs.device)
                res["nll_sum"] = -log_probs.gather(
                    1, targets[:, None]
                ).sum().item()

                if is_original:
                    # Argmax is stored separately, since float16 rounding can
                    # flip near-ties
                    entry["log_probs"] = log_probs.half().cpu()
                    entry["argmax"] = log_probs.argmax(-1).cpu()
                else:
                    cached = torch.load(cache_path)
                    if "log_probs" in cached:
                        orig_log_probs = cached["log_probs"].to(
                            log_probs.device
                        ).float()
                        res["kl_sum"] = (
                            orig_log_probs.exp()
                            * (orig_log_probs - log_probs)
                        ).sum().item()
                        res["agree_sum"] = (
                            cached["argmax"].to(log_probs.device)
                            == log_probs.argmax(-1)
                        ).sum().item()
                del logits, log_probs

        out = model.generate(
            ids_ten, attention_mask=torch.ones_like(ids_ten),
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=pad_id,
        )
        res["generation"] = tokenizer.decode(
            out[0, len(ids):], skip_special_tokens=True
        )
        results.append(res)

        if is_original:
            # Write atomically so an interrupted run never leaves a partial
            # entry behind
            entry["result"] = res
            torch.save(entry, cache_path + ".tmp")
            os.replace(cache_path + ".tmp", cache_path)

    num_nonfinite = sum(r["nonfinite"] for r in results)
    if num_nonfinite:
        print(
            "WARNING: %s produced non-finite logits on %d prompts (likely "
            "float16 overflow); these are excluded from the metrics"
            % (name, num_nonfinite)
        )
    return results


def _summarize(results, orig_results=None):
    """
    Aggregates per-prompt results into token-weighted metrics.
    """
    scored = [r for r in results if r["nll_sum"] is not None]
    num_tokens = sum(r["num_tokens"] for r in scored)
    summary = {
        "perplexity": (
            math.exp(sum(r["nll_sum"] for r in scored) / num_tokens)
            if num_tokens else None
        ),
        "num_prompts": len(results),
        "num_nonfinite": sum(r["nonfinite"] for r in results),
    }
    if orig_results is not None:
        compared = [r for r in results if r["kl_sum"] is not None]
        cmp_tokens = sum(r["num_tokens"] for r in compared)
        summary["kl"] = (
            sum(r["kl_sum"] for r in compared) / cmp_tokens
            if cmp_tokens else None
        )
        summary["top1_agreement"] = (
            sum(r["agree_sum"] for r in compared) / cmp_tokens
            if cmp_tokens else None
        )
        summary["exact_match"] = sum(
            r["generation"] == o["generation"]
            for r, o in zip(results, orig_results)
        ) / len(results)
    return summary


def _per_prompt(res):
    """
    Converts a raw per-prompt result into reportable per-prompt metrics.
    """
    out = {"generation": res["generation"]}
    if res["nll_sum"] is not None and res["num_tokens"]:
        out["perplexity"] = math.exp(res["nll_sum"] / res["num_tokens"])
    if res["kl_sum"] is not None:
        out["kl"] = res["kl_sum"] / res["num_tokens"]
        out["top1_agreement"] = res["agree_sum"] / res["num_tokens"]
    if res["nonfinite"]:
        out["nonfinite"] = True
    return out


def _fmt(val, spec):
    return "-" if val is None else format(val, spec)


@click.command()
@click.option(
    "--original", required=True,
    type=click.Path(exists=True, file_okay=False),
    help="path to the original (unpruned) model, used as the reference"
)
@click.option(
    "--pruned", required=True, multiple=True,
    type=click.Path(exists=True, file_okay=False),
    help="path to a pruned model; may be given multiple times"
)
@click.option(
    "--prompts", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="path to prompts file; can be .json (list of strings) or .txt (one "
    "string per line)"
)
@click.option(
    "--out-dir", required=True,
    help="directory to save summary.json and generations.jsonl"
)
@click.option(
    "--max-new-tokens", type=int, default=128,
    help="number of tokens to generate per prompt"
)
@click.option(
    "--max-prompt-length", type=int, default=1024,
    help="truncate prompts longer than this (in tokens)"
)
@click.option(
    "--limit", type=int, default=None,
    help="if specified, only use the first this many prompts"
)
@click.option(
    "--original-cache-dir", default=None,
    help="directory to cache the original model's results in, reused across "
    "runs; defaults to <out-dir>/original_cache"
)
@click.option(
    "--recompute-original", is_flag=True,
    help="if specified, clear the original model's cache and recompute it"
)
def main(
    original, pruned, prompts, out_dir, max_new_tokens, max_prompt_length,
    limit, original_cache_dir, recompute_original,
):
    """
    Evaluate pruned models against the original model on a prompts file.
    """
    prompt_list = _load_prompts(prompts)
    if limit is not None:
        prompt_list = prompt_list[:limit]

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        original, trust_remote_code=True,
    )
    input_ids_list = [
        encode_prompt(tokenizer, p)[:max_prompt_length] for p in prompt_list
    ]

    # Name models by directory, disambiguating duplicates
    names = []
    for path in pruned:
        name = os.path.basename(os.path.normpath(path))
        while name in names or name == "original":
            name += "_"
        names.append(name)

    os.makedirs(out_dir, exist_ok=True)
    cache_dir = original_cache_dir or os.path.join(out_dir, "original_cache")
    _prepare_cache(cache_dir, original, recompute_original)

    # Run the original model only on prompts that aren't cached yet
    keys = [_prompt_key(ids, max_new_tokens) for ids in input_ids_list]
    missing = [
        i for i, key in enumerate(keys)
        if not os.path.exists(os.path.join(cache_dir, key + ".pt"))
    ]
    print(
        "Evaluating %d prompts; original model results cached for %d/%d "
        "(%s)" % (
            len(keys), len(keys) - len(missing), len(keys), cache_dir
        )
    )
    if missing:
        missing_tokens = sum(
            max(len(input_ids_list[i]) - 1, 0) for i in missing
        )
        cache_gb = missing_tokens * len(tokenizer) * 2 / 1e9
        print("Caching original model results needs ~%.1f GB" % cache_gb)
        if cache_gb > 10:
            print(
                "WARNING: large log-prob cache; consider --limit or "
                "--max-prompt-length"
            )
        _run_model(
            original, "original", tokenizer,
            [input_ids_list[i] for i in missing], [keys[i] for i in missing],
            max_new_tokens, cache_dir, True,
        )

    all_results = {"original": [
        torch.load(os.path.join(cache_dir, key + ".pt"))["result"]
        for key in keys
    ]}
    for name, path in zip(names, pruned):
        all_results[name] = _run_model(
            path, name, tokenizer, input_ids_list, keys, max_new_tokens,
            cache_dir, False,
        )

    summary = {"original": _summarize(all_results["original"])}
    for name in names:
        summary[name] = _summarize(
            all_results[name], all_results["original"]
        )

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "generations.jsonl"), "w") as f:
        for prompt_i, prompt in enumerate(prompt_list):
            row = {"prompt": prompt}
            for name, results in all_results.items():
                row[name] = _per_prompt(results[prompt_i]) #type:ignore
            f.write(json.dumps(row) + "\n")

    name_width = max(len(n) for n in summary)
    print()
    print("%-*s  %10s  %8s  %8s  %8s  %9s" % (
        name_width, "model", "perplexity", "KL", "top1", "exact", "nonfinite"
    ))
    for name, s in summary.items():
        print("%-*s  %10s  %8s  %8s  %8s  %9d" % (
            name_width, name, _fmt(s["perplexity"], ".3f"),
            _fmt(s.get("kl"), ".4f"), _fmt(s.get("top1_agreement"), ".3f"),
            _fmt(s.get("exact_match"), ".3f"), s["num_nonfinite"],
        ))
    print("\nSaved summary.json and generations.jsonl to %s" % out_dir)


if __name__ == "__main__":
    main()
