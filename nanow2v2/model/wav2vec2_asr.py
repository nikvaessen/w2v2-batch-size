########################################################################################
#
# This class implements as wav2vec2 network for automatic speech recognition.
#
# Author(s): Nik Vaessen
########################################################################################

from dataclasses import dataclass
from typing import List, Optional

import torch.nn

from nanow2v2.model.wav2vec2 import Wav2vec2, Wav2vec2Config

########################################################################################
# Loss for ASR training


class CtcLoss(torch.nn.Module):
    def __init__(self, ctc_blank_idx: int = 0):
        super().__init__()

        self.blank_idx = ctc_blank_idx

    def forward(
        self,
        predictions: torch.Tensor,
        prediction_lengths: List[int],
        transcription_idx: torch.Tensor,
        transcription_lengths: List[int],
    ):
        original_device = predictions.device
        assert original_device == predictions.device == transcription_idx.device

        # predictions will be shape [BATCH_SIZE, MAX_INPUT_SEQUENCE_LENGTH, CLASSES]
        # expected to be [MAX_INPUT_SEQUENCE_LENGTH, BATCH_SIZE, CLASSES] for
        # loss function
        predictions = torch.transpose(predictions, 0, 1)

        # they also need to be log probabilities
        predictions = torch.nn.functional.log_softmax(predictions, dim=2)

        # prediction lengths will be shape [BATCH_SIZE]
        prediction_lengths = torch.tensor(prediction_lengths, dtype=torch.long)
        assert len(prediction_lengths.shape) == 1
        assert prediction_lengths.shape[0] == predictions.shape[1]

        # ground truths will be shape [BATCH_SIZE, MAX_TARGET_SEQUENCE_LENGTH]
        assert len(transcription_idx.shape) == 2
        assert transcription_idx.shape[0] == predictions.shape[1]

        # ground_truth_lengths will be shape [BATCH_SIZE]
        transcription_lengths = torch.tensor(transcription_lengths, dtype=torch.long)
        assert len(transcription_lengths.shape) == 1
        assert transcription_lengths.shape[0] == predictions.shape[1]

        # ctc loss expects every tensor to be on CPU
        # we disable cudnn due to variable input lengths
        # with torch.backends.cudnn.flags(enabled=False):
        return torch.nn.functional.ctc_loss(
            log_probs=predictions,
            targets=transcription_idx,
            input_lengths=prediction_lengths,
            target_lengths=transcription_lengths,
            blank=self.blank_idx,
            zero_infinity=True,  # prevents any weird crashes
        )


########################################################################################
# network implementation


@torch.no_grad()
def init_linear_for_ctc(linear: torch.nn.Linear):
    # set weight and bias such that prior for first class (blank token) is high

    # high bias for blank (idx 0)
    linear.bias[:1] += 5

    # weights for classes other than blank are 1 magnitude lower
    linear.weight[1:] *= 0.1


@dataclass
class ForASRConfig:
    vocab_size: int
    blank_idx: int

    freeze_cnn_for_initial_steps: int  # -1 always, 0 never,
    freeze_transformer_for_initial_steps: int  # -1 always, 0 never

    # only used when using huggingface implementation
    use_hf_ckpt: bool = False


class Wav2vec2ForASR(torch.nn.Module):
    def __init__(
        self, w2v2_cfg: Wav2vec2Config, asr_cfg: ForASRConfig, init_frozen: bool = True
    ):
        super().__init__()

        # save config
        self.w2v2_cfg = w2v2_cfg
        self.asr_cfg = asr_cfg

        # backbone model
        self.w2v2 = Wav2vec2(w2v2_cfg)

        # ASR head
        self.classifier = torch.nn.Linear(
            in_features=w2v2_cfg.num_dim_context, out_features=asr_cfg.vocab_size
        )
        self.ctc_loss = CtcLoss(asr_cfg.blank_idx)

        # linear predicts blank
        init_linear_for_ctc(self.classifier)

        # manage frozen state
        self.is_cnn_frozen = init_frozen
        self.is_transformer_frozen = init_frozen

        if init_frozen:
            self.toggle_grad_cnn(False)
            self.toggle_grad_transformer(False)

    def forward(self, raw_audio: torch.Tensor, sample_lengths: List[int]):
        context_features, new_sample_lengths = self.w2v2.forward(
            raw_audio, sample_lengths
        )
        vocab_prob = self.classifier(context_features)

        return vocab_prob, new_sample_lengths

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

        vocab_prob, new_sample_lengths = self.forward(raw_audio, sample_lengths)
        loss = self.ctc_loss.forward(
            vocab_prob, new_sample_lengths, transcription_idx, transcription_lengths
        )

        return loss, vocab_prob, new_sample_lengths

    def toggle_grad_cnn(self, requires_grad: bool):
        self.w2v2.conv_network.requires_grad_(requires_grad)
        self.w2v2.project_speech_feature.requires_grad_(requires_grad)
        self.w2v2.project_speech_feature_norm.requires_grad_(requires_grad)

    def toggle_grad_transformer(self, requires_grad: bool):
        self.w2v2.transformer_network.requires_grad_(requires_grad)
        self.w2v2.rel_pos_layer.requires_grad_(requires_grad)
        self.w2v2.masking_vector.requires_grad = requires_grad
