#! /usr/bin/env python3
########################################################################################
#
# Script to convert a fairseq checkpoint to a nanow2v2 checkpoint.
#
# Author(s): anon
########################################################################################

import pathlib

import click
import torch

########################################################################################
# main script


def convert_to_nano(network):
    prefix = "w2v2."
    out_ckpt = {}

    for k, v in network.items():
        has_prefix = True
        is_weight = k.split(".")[-1] == "weight"
        is_bias = k.split(".")[-1] == "bias"

        if is_weight and not is_bias:
            kind = "weight"
        elif is_bias and not is_weight:
            kind = "bias"
        else:
            kind = ""

        new_k = None

        if k == "mask_emb":
            new_k = "masking_vector"

        elif "feature_extractor" in k:
            layer = int(k.split(".")[2])
            if layer == 0:
                if "0.2" in k:
                    new_k = f"conv_network.conv_layers.{layer}.norm.{kind}"
                else:
                    new_k = f"conv_network.conv_layers.{layer}.conv.{kind}"
            else:
                new_k = f"conv_network.conv_layers.{layer}.conv.{kind}"

        elif "post_extract_proj" in k:
            new_k = f"project_speech_feature.{kind}"

        elif f"layer_norm.{kind}" == k:
            new_k = f"project_speech_feature_norm.{kind}"

        elif "encoder.pos_conv" in k:
            if is_bias:
                new_k = f"rel_pos_layer.conv.bias"
            elif "weight_g" in k:
                new_k = "rel_pos_layer.conv.weight_g"
            elif "weight_v" in k:
                new_k = "rel_pos_layer.conv.weight_v"

        elif "encoder.layers" in k:
            layer = int(k.split(".")[2])

            if "self_attn." in k:
                # special case because we model it as 1 tensor instead of 3
                if "k_proj" in k:
                    k_tensor = network[
                        f"encoder.layers.{layer}.self_attn.k_proj.{kind}"
                    ]
                    v_tensor = network[
                        f"encoder.layers.{layer}.self_attn.v_proj.{kind}"
                    ]
                    q_tensor = network[
                        f"encoder.layers.{layer}.self_attn.q_proj.{kind}"
                    ]
                    v = torch.cat([k_tensor, q_tensor, v_tensor], dim=0)
                    new_k = f"transformer_network.{layer}.attention.attention_projection.{kind}"

                elif "v_proj" in k or "q_proj" in k:
                    # we skip these as already done in k_proj
                    continue
                else:
                    # out_proj
                    new_k = (
                        f"transformer_network.{layer}.attention.out_projection.{kind}"
                    )

            elif "self_attn_layer_norm" in k:
                new_k = f"transformer_network.{layer}.norm_att.{kind}"
            elif "fc1" in k:
                new_k = f"transformer_network.{layer}.ffn.fc1.{kind}"
            elif "fc2" in k:
                new_k = f"transformer_network.{layer}.ffn.fc2.{kind}"
            elif "final_layer_norm" in k:
                new_k = f"transformer_network.{layer}.norm_ffn.{kind}"

        if "encoder.layer_norm" in k:
            new_k = f"rel_pos_layer.norm.{kind}"

        # self-supervised params
        # self-supervised params
        if "quantizer" in k:
            has_prefix = False
            if "vars" in k:
                new_k = "quantization_layer.quantization_choices"
                v = v.squeeze()
            else:
                new_k = f"quantization_layer.classification_layer.{kind}"

        if "project_q" in k:
            has_prefix = False
            new_k = f"project_quantized_feature.{kind}"

        if "final_proj" in k:
            has_prefix = False
            new_k = f"project_context_feature.{kind}"

        if new_k is None:
            raise ValueError(f"unhandled key {k}")

        out_ckpt[f"{prefix if has_prefix else ''}{new_k}"] = v.cpu()

    return out_ckpt


@click.command()
@click.argument("fairseq_ckpt", type=pathlib.Path)
@click.argument("out", type=pathlib.Path)
def main(fairseq_ckpt: pathlib.Path, out: pathlib.Path):
    """
    fairseq_ckpt -> path to ckpt from fairseq
    out -> path to save converted fairseq ckpt to
    """
    print("converting")
    print(f"fairseq_ckpt={str(fairseq_ckpt)}")
    print(f"out={str(out)}")

    ckpt_fairseq = torch.load(fairseq_ckpt)

    network_fairseq = ckpt_fairseq["model"]
    state_dict = convert_to_nano(network_fairseq)

    out.parent.mkdir(exist_ok=True, parents=True)
    torch.save({"network": state_dict}, str(out))


if __name__ == "__main__":
    main()
