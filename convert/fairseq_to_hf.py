#! /usr/bin/env python3
########################################################################################
#
# Script to convert a fairseq checkpoint to a huggingface checkpoint.
#
# Author(s): Nik Vaessen
########################################################################################

import pathlib
from typing import List

import click
import torch

########################################################################################
# main script


def convert_to_hf(network: dict, prefix: List[str]):
    out_ckpt = {}

    for k, v in network.items():
        is_weight = k.split(".")[-1] == "weight"
        is_bias = k.split(".")[-1] == "bias"
        prefix_list = list(prefix)

        if is_weight and not is_bias:
            kind = "weight"
        elif is_bias and not is_weight:
            kind = "bias"
        else:
            kind = ""

        new_k = None

        if k == "mask_emb":
            new_k = "masked_spec_embed"

        elif "feature_extractor" in k:
            layer = int(k.split(".")[2])
            if layer == 0:
                if "0.2" in k:
                    new_k = f"feature_extractor.conv_layers.0.layer_norm.{kind}"
                else:
                    new_k = f"feature_extractor.conv_layers.0.conv.{kind}"
            else:
                new_k = f"feature_extractor.conv_layers.{layer}.conv.{kind}"

        elif "post_extract_proj" in k:
            new_k = f"feature_projection.projection.{kind}"

        elif f"layer_norm.{kind}" == k:
            new_k = f"feature_projection.layer_norm.{kind}"

        elif "encoder.pos_conv" in k:
            if is_bias:
                new_k = f"encoder.pos_conv_embed.conv.bias"
            elif "weight_g" in k:
                new_k = "encoder.pos_conv_embed.conv.weight_g"
            elif "weight_v" in k:
                new_k = "encoder.pos_conv_embed.conv.weight_v"

        elif "encoder.layers" in k:
            layer = int(k.split(".")[2])

            if "self_attn." in k:
                matrix = k.split(".")[4]
                new_k = f"encoder.layers.{layer}.attention.{matrix}.{kind}"
            elif "self_attn_layer_norm" in k:
                new_k = f"encoder.layers.{layer}.layer_norm.{kind}"
            elif "fc1" in k:
                new_k = f"encoder.layers.{layer}.feed_forward.intermediate_dense.{kind}"
            elif "fc2" in k:
                new_k = f"encoder.layers.{layer}.feed_forward.output_dense.{kind}"
            elif "final_layer_norm" in k:
                new_k = f"encoder.layers.{layer}.final_layer_norm.{kind}"

        if "encoder.layer_norm" in k:
            new_k = f"encoder.layer_norm.{kind}"

        # self-supervised params
        if "quantizer" in k:
            del prefix_list[-1]
            if "vars" in k:
                new_k = "quantizer.codevectors"
            else:
                new_k = f"quantizer.weight_proj.{kind}"

        if "project_q" in k:
            del prefix_list[-1]
            new_k = k

        if "final_proj" in k:
            del prefix_list[-1]
            new_k = f"project_hid.{kind}"

        if new_k is None:
            raise ValueError(f"unhandled key {k}")

        if len(prefix_list) > 0:
            prefix_str = ".".join(prefix) + "."
        else:
            prefix_str = ""

        out_ckpt[f"{prefix_str}{new_k}"] = v.cpu()

    return out_ckpt


@click.command()
@click.argument("fairseq_ckpt", type=pathlib.Path)
@click.argument("out", type=pathlib.Path)
@click.option("--prefix", type=str, required=False, default="w2v2.wav2vec2")
def main(fairseq_ckpt: pathlib.Path, out: pathlib.Path, prefix: str):
    """
    fairseq_ckpt -> path to ckpt from fairseq
    out -> path to save converted fairseq ckpt to
    """
    print(f"converting with {prefix=}")
    print(f"fairseq_ckpt={str(fairseq_ckpt)}")
    print(f"out={str(out)}")

    prefix = prefix.split(".")
    assert prefix[-1] == "wav2vec2"

    ckpt_fairseq = torch.load(fairseq_ckpt)

    network_fairseq = ckpt_fairseq["model"]
    state_dict = convert_to_hf(network_fairseq, prefix)

    out.parent.mkdir(exist_ok=True, parents=True)
    torch.save({"network": state_dict}, str(out))


if __name__ == "__main__":
    main()
