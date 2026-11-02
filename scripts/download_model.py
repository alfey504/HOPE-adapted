"""
Downloads a HuggingFace MoE model and saves a local copy in the given dtype.

Useful for GPUs without native bfloat16 support (e.g. T4, P100, V100), where
running a bfloat16 checkpoint in float16 is considerably faster. The saved
directory can be passed directly as `--model-path` to `hope calibrate` and
`hope prune`.

Converting OLMoE-1B-7B requires ~14 GB of RAM, and ~2x the model size in disk
space (HuggingFace cache + output). On Kaggle, `/kaggle/working` is limited to
~20 GB, so write the output to `/kaggle/temp` or `/tmp` instead.

Usage:
    python scripts/download_model.py --model-id allenai/OLMoE-1B-7B-0924 \
        --out-path /path/to/olmoe-fp16 --dtype float16 --sanity-check
"""

import os
import click
import torch
import transformers


@click.command()
@click.option(
    "--model-id", default="allenai/OLMoE-1B-7B-0924",
    help="HuggingFace Hub model ID (or local path) to download"
)
@click.option(
    "--out-path", required=True,
    help="directory to save the converted model"
)
@click.option(
    "--dtype", default="float16",
    type=click.Choice(["float16", "bfloat16", "float32"]),
    help="dtype to save the model weights in"
)
@click.option(
    "--revision", default=None,
    help="optional model revision (branch, tag, or commit hash)"
)
@click.option(
    "--sanity-check", is_flag=True,
    help="if specified, run a short prompt through the saved model and check "
    "that the logits are finite"
)
def main(model_id, out_path, dtype, revision, sanity_check):
    """
    Download a HuggingFace model and save it in the given dtype.
    """
    torch_dtype = getattr(torch, dtype)

    print("Loading tokenizer and model from %s ..." % model_id)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, device_map="cpu",
        torch_dtype=torch_dtype, trust_remote_code=True,
    )

    os.makedirs(out_path, exist_ok=True)
    print("Saving %s model to %s ..." % (dtype, out_path))
    model.save_pretrained(out_path, save_original_format=True)
    tokenizer.save_pretrained(out_path)
    del model

    if sanity_check:
        _sanity_check(out_path, tokenizer)

    print("Done. Next step:")
    print(
        "    hope calibrate --model-path %s --prompts prompts.txt "
        "--out-path observations.h5" % out_path
    )


def _sanity_check(model_path, tokenizer):
    """
    Reloads the saved model, runs a short prompt through it, and checks that
    the logits are finite (float16 can overflow for bfloat16-trained models).
    """
    print("Running sanity check ...")
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device_map, torch_dtype="auto",
        trust_remote_code=True,
    )
    model.eval()

    ids = tokenizer(
        "The capital of France is", return_tensors="pt"
    ).input_ids.to(model.device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits
        if not torch.isfinite(logits).all():
            raise ValueError(
                "Non-finite logits from the saved model; it may overflow in "
                "this dtype. Try --dtype bfloat16 or float32."
            )
        out = model.generate(ids, max_new_tokens=10, do_sample=False)
    print("Sanity check passed. Sample: %r" % tokenizer.decode(out[0]))


if __name__ == "__main__":
    main()
