########################################################################################
#
# Wrap around HuggingFace's implementation of Wav2vec2 for self-supervised learning.
#
# Used as sanity check for own implementation.
#
# Author(s): Nik Vaessen
########################################################################################

from typing import List, Optional

import math

import numpy as np
import torch
import torch.nn as nn

from transformers import Wav2Vec2Config, Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import _sample_negative_indices

from nanow2v2.model.wav2vec2_ssl import (
    Wav2vec2ForSSL,
    ForSSLConfig,
    Wav2vec2Config,
    SSLForwardResult,
)

########################################################################################
# network


class HfWav2vec2ForSSL(Wav2vec2ForSSL):
    def __init__(self, w2v2_cfg: Wav2vec2Config, ssl_cfg: ForSSLConfig):
        super().__init__(w2v2_cfg, ssl_cfg)

        # delete everything from the super call
        del self.w2v2
        del self.quantization_layer
        del self.project_context_feature
        del self.project_quantized_feature
        del self.ssl_cfg

        # config
        self.w2v2_cfg = w2v2_cfg
        self.ssl_cfg = ssl_cfg

        # model
        self.w2v2 = Wav2Vec2ForPreTraining(
            Wav2Vec2Config(
                conv_dim=[self.w2v2_cfg.num_dim_speech] * 7,
                num_hidden_layers=self.w2v2_cfg.num_layers,
                hidden_size=self.w2v2_cfg.num_dim_context,
                num_attention_heads=self.w2v2_cfg.num_heads,
                intermediate_size=self.w2v2_cfg.num_dim_fnn,
                hidden_dropout=self.w2v2_cfg.dropout_prob,
                activation_dropout=self.w2v2_cfg.dropout_prob,
                attention_dropout=self.w2v2_cfg.dropout_prob,
                final_dropout=self.w2v2_cfg.dropout_prob,
                layerdrop=self.w2v2_cfg.layer_drop_prob,
                num_negatives=self.ssl_cfg.num_negative_samples,
                proj_codevector_dim=self.ssl_cfg.num_dim_similarity,
                diversity_loss_weight=self.ssl_cfg.diversity_loss_weight,
                contrastive_logits_temperature=self.ssl_cfg.contrastive_temperature,
                num_codevector_groups=self.ssl_cfg.num_codebooks,
                num_codevectors_per_group=self.ssl_cfg.num_entries,
                codevector_dim=self.ssl_cfg.num_dim_quantized,
                mask_time_prob=self.w2v2_cfg.percentage_masked,
                mask_time_length=self.w2v2_cfg.mask_span,
            )
        )

        self.w2v2 = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")

        if self.ssl_cfg.freeze_codebooks:
            self.w2v2.quantizer.codevectors.requires_grad_(False)

    def step_gumbel_temperature(self):
        self.w2v2.quantizer.temperature = np.clip(
            self.w2v2.quantizer.temperature * self.ssl_cfg.gumbel_temperature_factor,
            a_min=self.ssl_cfg.gumbel_temperature_floor,
            a_max=float("inf"),
        )

    def get_gumbel_temperature(self):
        return self.w2v2.quantizer.temperature

    def get_codebooks(self):
        return (
            self.w2v2.quantizer.codevectors,
            self.ssl_cfg.num_codebooks,
            self.ssl_cfg.num_entries,
        )

    def forward(
        self, raw_audio: torch.Tensor, sample_lengths: List[int]
    ) -> SSLForwardResult:
        sample_lengths = self.w2v2._get_feat_extract_output_lengths(
            torch.LongTensor(sample_lengths)
        ).tolist()

        attention_mask = create_attention_mask(sample_lengths, raw_audio.device)
        mask_time_idx = create_mask_time_idx(
            sample_lengths, self.w2v2_cfg.percentage_masked, self.w2v2_cfg.mask_span
        )
        sample_negative_idx = create_sample_negative_idx(
            sample_lengths, mask_time_idx, self.ssl_cfg.num_negative_samples
        )

        attention_mask = attention_mask.to(self.w2v2.device)
        mask_time_idx = mask_time_idx.to(self.w2v2.device)
        sample_negative_idx = sample_negative_idx.to(self.w2v2.device)

        output = _exposed_hf_ssl_forward(
            self=self.w2v2,
            input_values=raw_audio,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_idx,
            sampled_negative_indices=sample_negative_idx,
        )

        batch_size, seq_length, _ = output.speech_features.shape
        codebook_logits = output.codebook_logits.view(
            batch_size, seq_length, self.ssl_cfg.num_codebooks, self.ssl_cfg.num_entries
        )

        return SSLForwardResult(
            output.loss,
            output.loss_contrastive,
            output.loss_diversity,
            output.loss_l2,
            output.cpc_logits,
            output.cpc_targets,
            codebook_logits,
            output.speech_features,
            output.context_features,
            output.quantized_features,
            None,
            sample_lengths,
        )


def _exposed_hf_ssl_forward(
    self,
    input_values: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    mask_time_indices: Optional[torch.BoolTensor] = None,
    sampled_negative_indices: Optional[torch.BoolTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
) -> SSLForwardResult:
    # copied from huggingface, modified to return SslForward object
    if mask_time_indices is not None:
        mask_time_indices = mask_time_indices.to(torch.bool)

    outputs = self.wav2vec2(
        input_values,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        mask_time_indices=mask_time_indices,
        return_dict=True,
    )

    # 1. project all transformed features (including masked) to final vq dim
    context_features = outputs[0]
    transformer_features = self.project_hid(context_features)

    # 2. quantize all (unmasked) extracted features and project to final vq dim
    speech_features = outputs[1]
    extract_features = self.dropout_features(speech_features)

    if attention_mask is not None:
        # compute reduced attention_mask correponding to feature vectors
        attention_mask = self._get_feature_vector_attention_mask(
            extract_features.shape[1], attention_mask, add_adapter=False
        )

    quantized_features, codevector_perplexity = self.quantizer(
        extract_features, mask_time_indices=mask_time_indices
    )
    with torch.no_grad():
        codebook_logits = self.quantizer.weight_proj(extract_features)

    quantized_features = self.project_q(quantized_features)

    loss = contrastive_loss = diversity_loss = None
    if sampled_negative_indices is not None:
        batch_size, sequence_length, hidden_size = quantized_features.shape

        # for training, we sample negatives
        # 3. sample K negatives (distractors) quantized states for contrastive loss
        # if attention_mask is passed, make sure that padded feature vectors cannot be sampled
        # sample negative quantized vectors BTC => (BxT)C
        negative_quantized_features = quantized_features.view(-1, hidden_size)[
            sampled_negative_indices.long().view(-1)
        ]
        negative_quantized_features = negative_quantized_features.view(
            batch_size, sequence_length, -1, hidden_size
        ).permute(2, 0, 1, 3)

        # 4. compute logits, corresponding to `logs = sim(c_t, [q_t, \sim{q}_t]) / \kappa`
        # of equation (3) in https://arxiv.org/pdf/2006.11477.pdf
        logits = self.compute_contrastive_logits(
            quantized_features[None, :],
            negative_quantized_features,
            transformer_features,
            self.config.contrastive_logits_temperature,
        )

        # 5. if a negative vector is identical to the positive (i.e. when codebook utilization is low),
        # its cosine similarity will be masked
        neg_is_pos = (quantized_features == negative_quantized_features).all(-1)

        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")

        # 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
        # -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
        logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
        target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()

        contrastive_loss = nn.functional.cross_entropy(
            logits.float(), target, reduction="sum"
        )
        # 7. compute diversity loss: \mathbf{L}_d
        num_codevectors = (
            self.config.num_codevectors_per_group * self.config.num_codevector_groups
        )
        diversity_loss = (
            (num_codevectors - codevector_perplexity) / num_codevectors
        ) * mask_time_indices.sum()

        # 8. \mathbf{L} = \mathbf{L}_m + \alpha * \mathbf{L}_d
        loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss

    return SSLForwardResult(
        loss=loss,
        loss_contrastive=contrastive_loss,
        loss_diversity=diversity_loss,
        loss_l2=torch.tensor(-1),
        cpc_logits=logits,
        cpc_targets=target,
        codebook_logits=codebook_logits,
        speech_features=speech_features,
        context_features=context_features,
        quantized_features=quantized_features,
        mask_idx=None,
        sample_lengths=None,
    )


def create_attention_mask(sample_lengths: List[int], device: torch.device):
    mask = torch.zeros(
        (len(sample_lengths), max(sample_lengths)), dtype=torch.long, device=device
    )

    for idx, length in enumerate(sample_lengths):
        mask[idx, 0:length] = 1

    return mask


def create_mask_time_idx(
    sample_lengths: List[int],
    percentage_masked: float,
    mask_span: int,
    min_masks: int = 1,
):
    batch_size = len(sample_lengths)
    sample_length = max(sample_lengths)
    mask = torch.zeros((batch_size, sample_length), dtype=torch.bool)

    for bs in range(batch_size):
        # determine the amount of masks while making sure padded regions are ignored
        batch_length = max(mask_span, sample_lengths[bs] - mask_span)
        num_masks = max(
            min_masks, math.ceil(batch_length * percentage_masked / mask_span)
        )

        # randomly decide on the starting indexes of the masks
        random_idx = torch.randperm(batch_length)[0:num_masks]

        # add the values [0, ..., mask_span] to the random indexes
        span = torch.arange(0, mask_span).expand(num_masks, mask_span)
        mask_idx = (span + random_idx[:, None]).unique()

        # set the masked idx to true
        mask[bs, mask_idx] = True

    return mask


def create_sample_negative_idx(
    sample_lengths: List[int], mask_time: torch.Tensor, num_negatives: int
):
    batch_size = len(sample_lengths)
    sequence_length = max(sample_lengths)

    sampled_negative_indices = _sample_negative_indices(
        features_shape=(batch_size, sequence_length),
        num_negatives=num_negatives,
        mask_time_indices=mask_time.numpy(),
    )
    return torch.tensor(sampled_negative_indices)
