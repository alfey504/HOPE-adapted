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

Models are loaded one at a time, so only one needs to fit on the GPUs. The
original's next-token log-probabilities are cached on disk (float16, roughly
2 bytes x vocab size per prompt token) and deleted at the end. Because of the
float16 cache, KL values below ~1e-3 (including slightly negative values) are
within rounding error of 0.

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
import json
import math
import os
import shutil
import click
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


def _unload(model):
    """
    Frees a model's memory.
    """
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def _eval_model(
    model, tokenizer, input_ids_list, max_new_tokens, cache_dir, is_original,
    name,
):
    """
    Runs a model over all prompts. For the original model, caches next-token
    log-probs to `cache_dir`; for pruned models, compares against the cache.
    Returns a list of per-prompt result dictionaries.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    results = []
    for prompt_i, ids in enumerate(tqdm.tqdm(input_ids_list, desc=name)):
        ids_ten = torch.tensor([ids], device=model.device)
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

                cache_path = os.path.join(cache_dir, "%d.pt" % prompt_i)
                if is_original:
                    # Argmax is stored separately, since float16 rounding can
                    # flip near-ties
                    torch.save({
                        "log_probs": log_probs.half().cpu(),
                        "argmax": log_probs.argmax(-1).cpu(),
                    }, cache_path)
                elif os.path.exists(cache_path):
                    cached = torch.load(cache_path)
                    orig_log_probs = cached["log_probs"].to(
                        log_probs.device
                    ).float()
                    res["kl_sum"] = (
                        orig_log_probs.exp() * (orig_log_probs - log_probs)
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
def main(
    original, pruned, prompts, out_dir, max_new_tokens, max_prompt_length,
    limit,
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
    cache_dir = os.path.join(out_dir, "_orig_logprobs")
    os.makedirs(cache_dir, exist_ok=True)

    total_tokens = sum(max(len(ids) - 1, 0) for ids in input_ids_list)
    cache_gb = total_tokens * len(tokenizer) * 2 / 1e9
    print(
        "Evaluating %d prompts (%d tokens); log-prob cache needs ~%.1f GB"
        % (len(input_ids_list), total_tokens, cache_gb)
    )
    if cache_gb > 10:
        print(
            "WARNING: large log-prob cache; consider --limit or "
            "--max-prompt-length"
        )

    try:
        print("Loading original model from %s ..." % original)
        model = _load_model(original)
        all_results = {"original": _eval_model(
            model, tokenizer, input_ids_list, max_new_tokens, cache_dir,
            True, "original",
        )}
        _unload(model)

        for name, path in zip(names, pruned):
            print("Loading %s from %s ..." % (name, path))
            model = _load_model(path)
            all_results[name] = _eval_model(
                model, tokenizer, input_ids_list, max_new_tokens, cache_dir,
                False, name,
            )
            _unload(model)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

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
                row[name] = _per_prompt(results[prompt_i])
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
