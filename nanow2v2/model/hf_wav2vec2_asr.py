########################################################################################
#
# Wrap around HuggingFace's implementation of Wav2vec2 for ASR.
#
# Used as sanity check for own implementation.
#
# Author(s): Nik Vaessen
########################################################################################

from typing import List, Optional

import torch

from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

from nanow2v2.model.wav2vec2_asr import (
    init_linear_for_ctc,
    Wav2vec2ForASR,
    ForASRConfig,
    Wav2vec2Config,
)
from nanow2v2.model.hf_wav2vec2_ssl import create_attention_mask


########################################################################################
# model


class HfWav2vecForASR(Wav2vec2ForASR):
    def __init__(
        self, w2v2_cfg: Wav2vec2Config, asr_cfg: ForASRConfig, init_frozen: bool
    ):
        super().__init__(w2v2_cfg, asr_cfg, init_frozen=False)

        # overwrite model
        if self.asr_cfg.use_hf_ckpt:
            self.w2v2 = Wav2Vec2ForCTC.from_pretrained(self.asr_cfg.use_hf_ckpt)
        else:
            self.w2v2 = Wav2Vec2ForCTC(
                Wav2Vec2Config(
                    conv_dim=[self.w2v2_cfg.num_dim_speech] * 7,
                    num_hidden_layers=self.w2v2_cfg.num_layers,
                    hidden_size=self.w2v2_cfg.num_dim_context,
                    num_attention_heads=self.w2v2_cfg.num_heads,
                    intermediate_size=self.w2v2_cfg.num_dim_fnn,
                    hidden_dropout=self.w2v2_cfg.dropout_prob,
                    activation_dropout=self.w2v2_cfg.dropout_prob,
                    attention_dropout=self.w2v2_cfg.dropout_prob,
                    # final_dropout=self.w2v2_cfg.dropout_prob,
                    layerdrop=self.w2v2_cfg.layer_drop_prob,
                    mask_time_prob=self.w2v2_cfg.percentage_masked,
                    mask_time_length=self.w2v2_cfg.mask_span,
                    vocab_size=self.asr_cfg.vocab_size,
                    pad_token_id=self.asr_cfg.blank_idx,
                    # TODO remove hard-coding
                    # mask_feature_prob=0.1,
                    # mask_feature_length=64,
                    ctc_loss_reduction="mean",
                )
            )

        # delete classification head of super class
        del self.classifier
        del self.ctc_loss

        # predict blank by default
        # init_linear_for_ctc(self.w2v2.lm_head)

        # manage frozen state
        self.is_cnn_frozen = init_frozen
        self.is_transformer_frozen = init_frozen

        if init_frozen:
            self.toggle_grad_cnn(False)
            self.toggle_grad_transformer(False)

        self.w2v2.gradient_checkpointing_enable()

    def forward(self, raw_audio: torch.Tensor, sample_lengths: List[int]):
        """Simple forward pass for fine-tuning and/or inference"""
        sample_lengths_after_conv = self.w2v2._get_feat_extract_output_lengths(
            torch.LongTensor(sample_lengths)
        ).tolist()
        attention_mask = create_attention_mask(sample_lengths, raw_audio.device)

        output = self.w2v2(
            raw_audio, attention_mask=attention_mask, output_hidden_states=True
        )

        return output.logits, sample_lengths_after_conv

    def forward_asr_training(
        self,
        raw_audio: torch.Tensor,
        sample_lengths: List[int],
        transcription_idx: torch.Tensor,
        transcription_lengths: List[int],
        train_step: Optional[int] = None,
    ):
        if (
            train_step is not None
            and self.asr_cfg.freeze_cnn_for_initial_steps >= 0
            and self.is_cnn_frozen
            and train_step > self.asr_cfg.freeze_cnn_for_initial_steps
        ):
            self.toggle_grad_cnn(True)
            self.is_cnn_frozen = False

        if (
            train_step is not None
            and self.asr_cfg.freeze_transformer_for_initial_steps >= 0
            and self.is_transformer_frozen
            and train_step > self.asr_cfg.freeze_transformer_for_initial_steps
        ):
            self.toggle_grad_transformer(True)
            self.is_transformer_frozen = False

        attention_mask = create_attention_mask(sample_lengths, raw_audio.device)
        labels = create_labels_tensor(transcription_idx, transcription_lengths)
        sample_lengths_after_conv = self.w2v2._get_feat_extract_output_lengths(
            torch.LongTensor(sample_lengths)
        ).tolist()

        output = self.w2v2.forward(
            raw_audio,
            attention_mask=attention_mask,
            output_hidden_states=True,
            labels=labels,
        )

        return output.loss, output.logits, sample_lengths_after_conv

    def toggle_grad_cnn(self, requires_grad: bool):
        self.w2v2.wav2vec2.feature_extractor.requires_grad_(requires_grad)
        self.w2v2.wav2vec2.feature_projection.requires_grad_(requires_grad)

    def toggle_grad_transformer(self, requires_grad: bool):
        self.w2v2.wav2vec2.encoder.requires_grad_(requires_grad)
        self.w2v2.wav2vec2.masked_spec_embed.requires_grad = requires_grad


def create_labels_tensor(
    transcription_idx: torch.Tensor, transcription_lengths: List[int]
) -> torch.Tensor:
    for idx, length in enumerate(transcription_lengths):
        transcription_idx[idx, length:] = -100

    return transcription_idx
