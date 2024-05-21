#! /usr/bin/env python3
########################################################################################
#
# Script to convert a nanow2v2 checkpoint to a fairseq checkpoint.
#
# Author(s): anon
########################################################################################

import pathlib

import click
import torch

########################################################################################
# main script


def convert_to_fairseq(network):
    out_ckpt = {}

    for k, v in network.items():
        is_weight = k.split(".")[-1] == "weight"
        is_bias = k.split(".")[-1] == "bias"

        if is_weight and not is_bias:
            kind = "weight"
        elif is_bias and not is_weight:
            kind = "bias"
        else:
            kind = ""

        new_k = None

        if "masking_vector" in k:
            new_k = "mask_emb"

        elif "conv_network" in k:
            layer = int(k.split(".")[3])
            if layer == 0 and "norm" in k:
                new_k = f"feature_extractor.conv_layers.{layer}.2.{kind}"
            else:
                new_k = f"feature_extractor.conv_layers.{layer}.0.{kind}"

        elif f"project_speech_feature_norm.{kind}" in k:
            new_k = f"layer_norm.{kind}"

        elif "project_speech_feature." in k:
            new_k = f"post_extract_proj.{kind}"

        elif "rel_pos_layer" in k:
            if "conv" in k and is_bias:
                new_k = f"encoder.pos_conv.0.bias"
            elif "weight_g" in k:
                new_k = "encoder.pos_conv.0.weight_g"
            elif "weight_v" in k:
                new_k = "encoder.pos_conv.0.weight_v"
            elif "norm" in k:
                new_k = f"encoder.layer_norm.{kind}"

        elif "transformer_network" in k:
            layer = int(k.split(".")[2])

            if "attention" in k:
                # special case because we model it as 1 tensor instead of 3
                if "attention_projection" in k:
                    tensor = network[k]
                    num_dim = tensor.shape[0] // 3
                    assert tensor.shape[0] % 3 == 0

                    if kind == "weight":
                        k_tensor = tensor[num_dim * 0 : num_dim * 1, :].cpu().clone()
                        v_tensor = tensor[num_dim * 1 : num_dim * 2, :].cpu().clone()
                        q_tensor = tensor[num_dim * 2 : num_dim * 3, :].cpu().clone()

                        out_ckpt[
                            f"encoder.layers.{layer}.self_attn.k_proj.weight"
                        ] = k_tensor
                        out_ckpt[
                            f"encoder.layers.{layer}.self_attn.v_proj.weight"
                        ] = v_tensor
                        out_ckpt[
                            f"encoder.layers.{layer}.self_attn.q_proj.weight"
                        ] = q_tensor
                    else:
                        k_tensor = tensor[num_dim * 0 : num_dim * 1].cpu().clone()
                        v_tensor = tensor[num_dim * 1 : num_dim * 2].cpu().clone()
                        q_tensor = tensor[num_dim * 2 : num_dim * 3].cpu().clone()
                        out_ckpt[
                            f"encoder.layers.{layer}.self_attn.k_proj.bias"
                        ] = k_tensor
                        out_ckpt[
                            f"encoder.layers.{layer}.self_attn.v_proj.bias"
                        ] = v_tensor
                        out_ckpt[
                            f"encoder.layers.{layer}.self_attn.q_proj.bias"
                        ] = q_tensor

                    continue
                else:
                    # out_proj
                    new_k = f"encoder.layers.{layer}.self_attn.out_proj.{kind}"

            elif "norm_att" in k:
                new_k = f"encoder.layers.{layer}.self_attn_layer_norm.{kind}"
            elif "fc1" in k:
                new_k = f"encoder.layers.{layer}.fc1.{kind}"
            elif "fc2" in k:
                new_k = f"encoder.layers.{layer}.fc2.{kind}"
            elif "norm_ffn" in k:
                new_k = f"encoder.layers.{layer}.final_layer_norm.{kind}"

        if "encoder.layer_norm" in k:
            new_k = f"rel_pos_layer.norm.{kind}"

        # self-supervised params
        if "quantization_layer" in k:
            if "classification_layer" in k:
                new_k = f"quantizer.weight_proj.{kind}"
            elif "quantization_choices" in k:
                new_k = "quantizer.vars"
                v = v[None, :, :]
            elif "temp" in k:
                continue

        if "project_quantized_feature" in k:
            new_k = f"project_q.{kind}"

        if "project_context_feature" in k:
            new_k = f"final_proj.{kind}"

        if new_k is None:
            raise ValueError(f"unhandled key {k}")

        out_ckpt[f"{new_k}"] = v.cpu()

    return out_ckpt


@click.command()
@click.argument("ckpt", type=pathlib.Path)
@click.argument("existing_fairseq_ckpt", type=pathlib.Path)
@click.argument("out", type=pathlib.Path)
def main(ckpt: pathlib.Path, existing_fairseq_ckpt: pathlib.Path, out: pathlib.Path):
    """
    fairseq_ckpt -> path to ckpt from fairseq
    out -> path to save converted fairseq ckpt to
    """
    print("converting")
    print(f"ckpt={str(ckpt)}")
    print(f"out={str(out)}")

    ckpt = torch.load(ckpt)

    network = ckpt["network"]
    state_dict = convert_to_fairseq(network)

    existing_ckpt = torch.load(existing_fairseq_ckpt)
    existing_ckpt["model"] = state_dict

    out.parent.mkdir(exist_ok=True, parents=True)
    torch.save(existing_ckpt, str(out))


if __name__ == "__main__":
    main()
