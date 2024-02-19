"""
This file contains a full implementation of wav2vec2. The implementation
tries to be both educational and fast.

The reference implementation is found in fairseq and huggingface:
1) fairseq -
2) huggingface -

This implementation is also inspired by nanoGPT: https://github.com/karpathy/nanoGPT.

We make a few changes to improve running time:
1. LayerNorm with weights but without bias (see https://arxiv.org/abs/1911.07013)
2. FlashAttention (see https://arxiv.org/abs/2205.14135)

Author: Nik Vaessen
"""

import math
import random
import pathlib

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

########################################################################################
# layers


class LayerNorm(nn.Module):
    """
    LayerNorm with an optional bias.
    """

    def __init__(self, num_dim: int, use_bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_dim))

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(num_dim))
        else:
            # we store bias for checkpoint compatability
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        # x has shape [BATCH_SIZE, SEQUENCE_LENGTH, NUM_DIM)
        # Each of the BATCH_SIZE*SEQUENCE_LENGTH vectors `v` of size NUM_DIM
        # will *independently* be normalized to unit mean and variance
        # by (v - t.mean(v)) / (t.var(v) + 1e-5)
        # before scalar multiplication with self.weight
        # and optional scaler addition with self.bias
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias)


class SelfAttention(nn.Module):
    """
    The (multi-head) self-attention layer with flash attention.
    """

    def __init__(
        self, num_dim: int, num_heads: int, dropout_prob: float, use_bias: bool
    ):
        super().__init__()
        assert num_dim % num_heads == 0

        # the key, value and query projection as a single (batched) layer
        self.attention_projection = nn.Linear(num_dim, num_dim * 3, bias=use_bias)
        nn.init.xavier_uniform_(self.attention_projection.weight, gain=1 / math.sqrt(2))

        # logic for multi-head attention
        self.num_dim = num_dim
        self.num_heads = num_heads

        self.head_dim = num_dim // num_heads

        # the projection applied after the concatenation of the output of
        # scaled dot-product attention
        self.out_projection = nn.Linear(num_dim, num_dim, bias=use_bias)
        nn.init.xavier_uniform_(self.out_projection.weight)

        # regularization
        self.dropout_prob = dropout_prob
        self.out_dropout = nn.Dropout(p=dropout_prob)

        if use_bias:
            nn.init.constant_(self.out_projection.bias, 0)
            nn.init.constant_(self.attention_projection.bias, 0)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # x has shape [BATCH_SIZE, SEQUENCE_LENGTH, NUM_DIM]
        # attention mask has shape [BATCH_SIZE, NUM_HEADS, SEQ_LENGTH, SEQ_LENGTH]
        bs, seq_length, num_dim = x.shape

        # we first compute the key/query/value output
        k, q, v = self.attention_projection(x).split(self.num_dim, dim=2)

        # we then split the output into heads with shape
        # [BATCH_SIZE, SEQUENCE_LENGTH, NUM_HEADS, NUM_DIM_HEAD]
        k = k.view(bs, seq_length, self.num_heads, self.head_dim)
        q = q.view(bs, seq_length, self.num_heads, self.head_dim)
        v = v.view(bs, seq_length, self.num_heads, self.head_dim)

        # We transpose the key/query/value to shape
        # [BATCH_SIZE, NUM_HEADS, SEQUENCE_LENGTH, NUM_DIM_HEAD]
        #  so that we easily compute operations for each head separately
        k, q, v = k.transpose(1, 2), q.transpose(1, 2), v.transpose(1, 2)

        # now we can apply self-attention for each head such that
        # `y ~= softmax( (QK^T) / sqrt(num_dim) )* V`
        # note the attention mask ensures feature vectors from padded time steps are
        # ignored as their attention score is set to -inf
        y = F.scaled_dot_product_attention(
            key=k,
            query=q,
            value=v,
            attn_mask=attention_mask,
            dropout_p=self.dropout_prob,
            is_causal=False,
        )

        # we concatenate the heads, so that we go back to the original shape
        y = y.transpose(1, 2).contiguous().view(bs, seq_length, num_dim)

        # we apply the final projection and dropout
        y = self.out_projection(y)
        y = self.out_dropout(y)

        return y


class FeedForwardNetwork(nn.Module):
    """
    The feed-forward network of a transformer block.
    """

    def __init__(
        self, num_dim: int, hidden_dim: int, dropout_prob: float, use_bias: bool
    ):
        super().__init__()

        self.fc1 = nn.Linear(num_dim, hidden_dim, bias=use_bias)
        self.fc2 = nn.Linear(hidden_dim, num_dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout_prob)

        if use_bias:
            nn.init.constant_(self.fc1.bias, 0)
            nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x: torch.Tensor):
        # x has shape [BATCH_SIZE, SEQUENCE_LENGTH, NUM_DIM]

        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class TransformerBlock(nn.Module):
    """
    A transformer block/layer, which consists of multi-headed self-attention and a
    feed-forward network.
    """

    def __init__(
        self,
        num_dim: int,
        num_dim_ffn: int,
        num_heads: int,
        dropout_prob: float,
        use_bias: bool,
    ):
        super().__init__()
        self.attention = SelfAttention(num_dim, num_heads, dropout_prob, use_bias)
        self.norm_att = LayerNorm(num_dim, use_bias)

        self.ffn = FeedForwardNetwork(num_dim, num_dim_ffn, dropout_prob, use_bias)
        self.norm_ffn = LayerNorm(num_dim, use_bias)

    def forward(self, x: torch.Tensor, attention_mask):
        # x has shape [BATCH_SIZE, SEQUENCE_LENGTH, NUM_DIM]
        # mask has shape [BATCH_SIZE, NUM_HEADS, SEQUENCE_LENGTH, SEQUENCE_LENGTH]

        x = self.norm_att(x + self.attention(x, attention_mask))
        x = self.norm_ffn(x + self.ffn(x))

        return x


class ConvolutionalLayer(nn.Module):
    """
    The convolutional layer of the CNN network, which consists of conv1d+layernorm+gelu
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int,
        padding: int,
        dilation: int,
        use_bias: bool,
        norm_type: Optional[str] = None,
    ):
        super().__init__()

        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel, stride, padding, dilation, bias=use_bias
        )
        nn.init.kaiming_normal_(self.conv.weight)

        self.norm_type = norm_type
        if self.norm_type is None:
            pass
        elif self.norm_type in ["layer", "group"]:
            self.norm = (
                LayerNorm(out_channels, use_bias)
                if norm_type == "layer"
                else torch.nn.GroupNorm(out_channels, out_channels)
            )
        else:
            raise ValueError(f"unknown {norm_type=}")

    def forward(self, x: torch.Tensor):
        # x has shape [BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH]
        x = self.conv(x)

        # we normalize over the channel dimension
        if self.norm_type == "layer":
            # For LayerNorm, channels need to be the last dimension
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        elif self.norm_type == "group":
            # For GroupNorm, channels need to be the second dimension
            x = self.norm(x)

        x = F.gelu(x, approximate="tanh")
        return x

    def dim_out(self, dim_in: int):
        # output dimension as given in formula found at
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        num = dim_in + (2 * self.padding) - (self.dilation * (self.kernel - 1)) - 1
        den = self.stride

        dim_out = math.floor((num / den) + 1)

        return dim_out


class ConvolutionalNetwork(nn.Module):
    """
    The CNN network which encodes raw audio waveforms into feature representations.
    """

    def __init__(
        self,
        num_dim_speech: int,
        use_bias: bool,
        norm_type_first_layer: Optional[str] = None,
        norm_type_remaining_layers: Optional[str] = None,
    ):
        super().__init__()
        channels = num_dim_speech

        first_norm_type = norm_type_first_layer
        norm_type = norm_type_remaining_layers

        self.conv_layers = nn.ModuleList(
            [
                # first convolution has large kernel (10) and stride (5)
                ConvolutionalLayer(1, channels, 10, 5, 3, 1, use_bias, first_norm_type),
                # the next 4 layers have smaller kernel (3) and stride (2)
                ConvolutionalLayer(channels, channels, 3, 2, 1, 1, use_bias, norm_type),
                ConvolutionalLayer(channels, channels, 3, 2, 1, 1, use_bias, norm_type),
                ConvolutionalLayer(channels, channels, 3, 2, 1, 1, use_bias, norm_type),
                ConvolutionalLayer(channels, channels, 3, 2, 1, 1, use_bias, norm_type),
                # the last 2 layers have the smallest kernel (2), but same stride (2)
                ConvolutionalLayer(channels, channels, 2, 2, 0, 1, use_bias, norm_type),
                ConvolutionalLayer(channels, channels, 2, 2, 0, 1, use_bias, norm_type),
            ]
        )

    def forward(self, x: torch.Tensor, sample_lengths: List[int]):
        # x has shape [BATCH_SIZE, NUM_AUDIO_FRAMES]
        x = x[:, None, :]  # conv1 wants [BATCH_SIZE, NUM_CHANNELS, NUM_AUDIO_FRAMES]

        # Each channel should be seen as a waveform.
        # The first conv layer maps the (single) raw audio waveform to a stack of 512
        # (assuming num_dim_speech=512) shorter waveforms (512 different kernels).
        # These 512 waveforms are then further processed with the next 6 conv layers.
        for layer in self.conv_layers:
            x = layer(x)

            # compute the new sample lengths (which were reduced due to the convolution)
            sample_lengths = [layer.dim_out(s) for s in sample_lengths]

        # we will return shape [BATCH_SIZE, SEQUENCE_LENGTH, NUM_DIM_SPEECH].
        # This means we create a feature vector of the speech at timestep t by simply
        # stacking the values of each of the 512 waveforms at timestep t.
        return x.transpose(1, 2), sample_lengths


class RelativePositionLayer(nn.Module):
    """
    Compute an additive relative positional embedding based on the speech features.
    """

    def __init__(self, num_dim: int, use_bias: bool):
        super().__init__()

        self.conv = nn.Conv1d(num_dim, num_dim, kernel_size=128, padding=64, groups=16)
        nn.init.normal_(self.conv.weight, mean=0, std=math.sqrt(4 / (num_dim * 128)))
        nn.init.constant_(self.conv.bias, 0)

        torch.nn.utils.weight_norm(self.conv, dim=2)

        self.norm = LayerNorm(num_dim, use_bias)

    def forward(self, x: torch.Tensor):
        # x has shape [BATCH_SIZE, SEQUENCE_LENGTH, NUM_DIM]
        # conv1 wants [BATCH_SIZE, NUM_CHANNELS=NUM_DIM, SEQUENCE_LENGTH]
        rpe = x.transpose(1, 2)

        rpe = self.conv(rpe)
        rpe = F.gelu(rpe, approximate="tanh")
        rpe = rpe.transpose(1, 2)

        # ensure same sequence length as input
        if rpe.shape[1] > x.shape[1]:
            rpe = rpe[:, 0 : x.shape[1], :]

        # add rpe to input
        x = x + rpe
        x = self.norm(x)

        return x


########################################################################################
# network implementation


@dataclass
class Wav2vec2Config:
    # whether to use bias in parts of network
    use_bias_in_cnn: bool = False
    use_bias_in_proj: bool = True
    use_bias_in_transformer: bool = True

    # CNN settings
    use_mfcc: bool = False
    num_dim_speech: int = 512
    cnn_norm_type_first_layer: Optional[str] = "group"
    cnn_norm_type_remaining_layers: Optional[str] = None

    # transformer settings
    num_layers: int = 12
    num_heads: int = 12
    num_dim_context: int = 768
    num_dim_fnn: int = num_dim_context * 4

    # regularization
    dropout_prob: float = 0.1
    layer_drop_prob: float = 0.0

    # mask settings
    # acts as regularisation during fine-tuning,
    # part of training objective during self-supervised pre-training
    percentage_masked: float = 0.5  # how much time steps should be masked (lower bound)
    mask_span: int = 10  # the size of individual masks

    # optional checkpoint to load from
    init_ckpt: Optional[pathlib.Path] = None


class Wav2vec2(nn.Module):
    def __init__(self, cfg: Wav2vec2Config):
        super().__init__()
        self.cfg = cfg

        # Computes speech representations from raw audio
        self.conv_network = ConvolutionalNetwork(
            cfg.num_dim_speech,
            cfg.use_bias_in_cnn,
            cfg.cnn_norm_type_first_layer,
            cfg.cnn_norm_type_remaining_layers,
        )

        # Projects output of CNN to correct dimension for transformer network
        self.project_speech_feature_norm = LayerNorm(
            cfg.num_dim_speech, cfg.use_bias_in_proj
        )
        self.project_speech_feature = nn.Linear(
            cfg.num_dim_speech, cfg.num_dim_context, bias=cfg.use_bias_in_proj
        )

        # Computes positional embedding which is added to the speech representation
        self.rel_pos_layer = RelativePositionLayer(
            cfg.num_dim_context, cfg.use_bias_in_transformer
        )

        # time steps will be masked by replacing it with this (learned) vector
        self.masking_vector = nn.Parameter(
            torch.rand((cfg.num_dim_context,)), requires_grad=True
        )

        # Computes context representations based on speech representations
        self.transformer_network = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.num_dim_context,
                    cfg.num_dim_fnn,
                    cfg.num_heads,
                    cfg.dropout_prob,
                    cfg.use_bias_in_transformer,
                )
                for _ in range(cfg.num_layers)
            ]
        )

    def forward(self, raw_audio: torch.Tensor, sample_lengths: List[int]):
        """Simple forward pass for fine-tuning and/or inference"""
        # speech features from raw audio
        speech_features, sample_lengths = self.speech_features(
            raw_audio, sample_lengths
        )

        # context features with transformers
        context_features, mask = self.context_features(speech_features, sample_lengths)

        return context_features, sample_lengths

    def speech_features(
        self, raw_audio: torch.Tensor, sample_lengths: List[int]
    ) -> Tuple[torch.Tensor, List[int]]:
        # input has shape [BATCH_SIZE, NUM_AUDIO_FRAMES]
        return self.conv_network(raw_audio, sample_lengths)

    def context_features(
        self,
        speech_features: torch.Tensor,
        sample_lengths: List[int],
        require_mask: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # input has shape [BATCH_SIZE, SEQ_LENGTH, NUM_SPEECH_FEATURES]

        # scale to [BATCH_SIZE, SEQ_LENGTH, NUM_DIM]
        speech_features = self.project_speech_feature_norm(speech_features)
        speech_features = self.project_speech_feature(speech_features)

        # mask some time steps in the speech features to mimic specaugment
        # and to enable self-supervised learning
        if self.training or require_mask:
            speech_features, masked_idx_per_batch = mask_speech_features(
                speech_features,
                self.masking_vector,
                sample_lengths,
                self.cfg.percentage_masked,
                self.cfg.mask_span,
            )
        else:
            masked_idx_per_batch = None

        # add relative positional embedding
        context_features = self.rel_pos_layer(speech_features)

        # self-attention mask due to padding in batch, expanded to num attention heads
        self_attention_mask = construct_self_attention_mask(
            sample_lengths, context_features.device, self.cfg.num_heads
        )

        # compute context features with transformer
        for layer in self.transformer_network:
            if random.random() < self.cfg.layer_drop_prob:
                continue

            context_features = layer(context_features, self_attention_mask)

        return context_features, masked_idx_per_batch


########################################################################################
# utility for network


def construct_self_attention_mask(
    sample_lengths: List[int], device: torch.device, num_heads: int
) -> torch.Tensor:
    """
    Construct the mask which will be used in self-attention to ignore feature vectors
    at time steps which were padded to make every sample in the batch equal length.
    """
    bs = len(sample_lengths)
    seq_length = max(sample_lengths)
    mask = torch.zeros(
        (bs, 1, seq_length, seq_length),
        dtype=torch.bool,
        device=device,
        requires_grad=False,
    )

    for batch_idx, sample_length in enumerate(sample_lengths):
        mask[batch_idx, :, :, 0:sample_length] = True

    mask = mask.expand(-1, num_heads, -1, -1)

    return mask


def mask_speech_features(
    speech_features: torch.Tensor,
    masking_vector: torch.Tensor,
    sample_lengths: List[int],
    percentage_masked: float,
    mask_span: int,
    min_masks: int = 1,
):
    """
    Implement the masking of certain regions of the input to enable
    self-supervised learning, as well as regularization during fine-tuning.
    """
    batch_size, seq_length, num_dim = speech_features.shape
    device = speech_features.device

    # In order to mas certain time steps of the speech features, we create 2 masks:
    # The first is a mask of 0/1 values, with we multiply with the speech features.
    # The second is contains the replacement vector, which we will add instead.
    zeroing_mask = torch.ones_like(
        speech_features, dtype=torch.bool, device=device, requires_grad=False
    )
    zero_vector = torch.zeros(
        (num_dim,), dtype=zeroing_mask.dtype, device=device, requires_grad=False
    )

    # we also store which indexes have been masked for each batch
    mask_idx_per_batch = []

    # We will scatter the correct values to the masks for each sample in the batch
    with torch.no_grad():
        for bs in range(batch_size):
            # determine the amount of masks while making sure padded regions are ignored
            batch_length = max(mask_span, sample_lengths[bs] - mask_span)
            num_masks = max(
                min_masks, math.ceil(batch_length * percentage_masked / mask_span)
            )

            # randomly decide on the starting indexes of the masks
            random_idx = torch.randperm(batch_length, device=device)[0:num_masks]

            # add the values [0, ..., mask_span] to the random indexes
            span = torch.arange(0, mask_span, device=device).expand(
                num_masks, mask_span
            )
            mask_idx = (span + random_idx[:, None]).unique()

            mask_idx_per_batch.append(mask_idx)
            zeroing_mask[bs, mask_idx, :] = zero_vector

    # we first zero out every timestep which is masked by multiplying with 0
    speech_features = speech_features * zeroing_mask

    # and then we replace with the masking vector
    vector_mask = masking_vector.expand(batch_size, seq_length, num_dim)
    vector_mask = vector_mask * ~zeroing_mask
    speech_features = speech_features + vector_mask

    return speech_features, mask_idx_per_batch
