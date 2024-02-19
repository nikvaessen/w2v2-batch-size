########################################################################################
#
# This class implements as wav2vec2 network for automatic speech recognition.
#
# Author(s): Nik Vaessen
########################################################################################

from dataclasses import dataclass
from typing import List, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanow2v2.model.wav2vec2 import Wav2vec2, Wav2vec2Config

########################################################################################
# Quantization layer needed for SSL


class QuantizationLayer(nn.Module):
    """Quantize speech features during self-supervised training"""

    def __init__(
        self,
        num_dim_speech: int,
        num_codebooks: int,
        num_entries: int,
        output_dim: int,
        temp: float,
        n_chunked_init: int = 0,
    ):
        super().__init__()
        assert output_dim % num_codebooks == 0
        self.num_codebooks = num_codebooks
        self.num_entries = num_entries
        self.entry_dim = output_dim // num_codebooks

        # `G` codebooks with `v` entries of shape [entry_dim] are stored
        # as a single matrix
        self.quantization_choices = nn.Parameter(
            torch.FloatTensor(num_codebooks * num_entries, self.entry_dim),
            requires_grad=True,
        )
        if n_chunked_init > 0:
            self._init_chunked_codebook(n_chunked_init)
        else:
            nn.init.uniform_(self.quantization_choices)

        # classification layer mapping speech features to a discrete unit
        # in each codebook
        self.classification_layer = nn.Linear(
            num_dim_speech, num_codebooks * num_entries
        )
        nn.init.normal_(self.classification_layer.weight, mean=0, std=1)
        nn.init.zeros_(self.classification_layer.bias)

        # temperature of gumbel-softmax (lower=approx argmax, higher=uniformly random)
        self.temp = nn.Parameter(
            torch.tensor(temp, dtype=torch.float), requires_grad=False
        )

    def forward(self, x: torch.Tensor):
        # input has shape [BATCH_SIZE, SEQUENCE_LENGTH, NUM_DIM_SPEECH]
        batch_size, seq_length, num_dim = x.shape

        # We want to map each speech vector of size NUM_DIM_SPEECH to a quantized
        # vector of size ENTRY_DIM*NUM_CODEBOOKS.
        # For each speech feature vector, and for each codebook,
        # we compute the probability of picking a specific entry of the codebook.
        logits_batch = self.classification_layer(x)

        # view as shape [NUM_SPEECH_VECTORS*NUM_CODEBOOKS, NUM_CODEBOOK_ENTRIES]
        # so it's easy to apply softmax/argmax over logits
        logits = logits_batch.view(
            batch_size * seq_length * self.num_codebooks, self.num_entries
        )

        if self.training:
            # gumbel softmax will return a probability of 1 for a single entry
            # while still being differentiable (compared to argmax which isn't)
            probs = F.gumbel_softmax(logits, tau=self.temp, hard=True, dim=1)
        else:
            # during validation/inference we can simply use argmax
            idx = torch.argmax(logits, dim=1, keepdim=True)

            # we reconstruct the index as a one-hot vector of probabilities
            probs = torch.zeros_like(logits)
            probs.scatter_(dim=1, index=idx, value=1.0)

        # based on the (one-hot) probabilities, we can select the codebook entries
        # by applying a weighted-sum of the entries with probabilities as weights.
        # We do it this way because we still have to avoid argmax.

        # view probs such that last dimension is equal to number of codebook vectors
        probs = probs.view(
            batch_size * seq_length, self.num_codebooks * self.num_entries
        )

        # now we can do a weighted-sum
        q_per_codebook = probs[:, :, None] * self.quantization_choices

        # we reshape, so we can sum over the codebooks (all but one are 0-valued)
        q_per_codebook = q_per_codebook.view(
            batch_size * seq_length,
            self.num_codebooks,
            self.num_entries,
            self.entry_dim,
        )
        q_per_codebook = q_per_codebook.sum(dim=2)

        # we concat the entries from each notebook collapsing the last 2 dimension
        q = q_per_codebook.view(batch_size, seq_length, -1)

        # shape logits such that it is easier to reason about codebooks and entries
        logits = logits.view(
            batch_size, seq_length, self.num_codebooks, self.num_entries
        )

        # we return a shape [BATCH_SIZE, SEQ_LENGTH, NUM_CODEBOOKS*ENTRIES_DIM] for q
        return q, logits


########################################################################################
# loss for SSL


def _sample_negatives(
    sample_lengths: List[int],
    mask_idx: List[torch.Tensor],
    num_negative_samples: int,  # this is an upper-bound due to overlap
    device: torch.device,
    only_masked_negatives: bool = True,
):
    batch_size = len(mask_idx)

    neg_idxes = []
    for bs in range(batch_size):
        seq_length = sample_lengths[bs]
        m_idx = mask_idx[bs]

        # sample negatives for each timestep (some repeats are possible)
        if only_masked_negatives:
            neg_idx = torch.randint(
                0, len(m_idx), (seq_length, num_negative_samples), device=device
            )
            neg_idx = m_idx[neg_idx]
        else:
            neg_idx = torch.randint(
                0, seq_length, (seq_length, num_negative_samples), device=device
            )

        neg_idxes.append(neg_idx)

    return neg_idxes


def contrastive_loss(
    quantized_features: torch.Tensor,
    context_features: torch.Tensor,
    mask_idx: List[torch.Tensor],
    sample_lengths: List[int],
    num_negative_samples: int = 100,
    temperature: float = 0.1,
    ce_reduction: str = "mean",
    apply_cpc_over_masked_regions: bool = False,
    sample_neg_from_masked_regions_only: bool = False,
):
    batch_size, _, _ = quantized_features.shape
    device = quantized_features.device
    assert len(sample_lengths) == len(mask_idx) == batch_size

    # determine all negative indexes
    negative_idxes = _sample_negatives(
        sample_lengths,
        mask_idx,
        num_negative_samples,
        device,
        only_masked_negatives=sample_neg_from_masked_regions_only,
    )

    logits_per_batch = []
    for bs in range(batch_size):
        seq_length = sample_lengths[bs]

        # get the corresponding vectors as [SEQUENCE_LENGTH, NUM_NEGATIVE, SIM_DIM]
        neg_idx = negative_idxes[bs]
        num_negative_samples = neg_idx.shape[1]
        negatives = quantized_features[bs, neg_idx, :]

        # add the positive as the first index
        positives = quantized_features[bs, :seq_length, :][:, None, :]

        # we now create 'targets' of shape [NUM_NEGATIVE+1, SEQ_LENGTH, SIM_DIM]
        # where the first index will be positive sample
        targets = torch.cat([positives, negatives], dim=1)
        targets = targets.transpose(0, 1)

        # compute the cosine similarity between the targets and the prediction, which
        # is the context network output of timestep t
        prediction = context_features[bs, :seq_length, :]

        # we compute "logits" between the prediction and the target by using
        # cosine similarity
        logits = F.cosine_similarity(prediction, targets, dim=-1)

        # we transpose the logits to shape [SEQ_LENGTH, NUM_NEG], so we
        # can see it as a batch of size SEQ_LENGTH with NUM_NEG+1 class predictions
        logits = logits.transpose(0, 1)

        # our negative indexes have the problem that the negative index can be equal to
        # the positive index. We need to set the logit to -inf wherever this happens
        pos_idx = torch.arange(seq_length, device=device)[:, None].expand(
            (seq_length, num_negative_samples)
        )

        pos_is_neg = pos_idx == neg_idx

        # add a 'false' row which is the actual logit for the positive class
        pos_is_neg = torch.cat(
            [torch.zeros((seq_length, 1), dtype=torch.bool, device=device), pos_is_neg],
            dim=1,
        )

        logits[pos_is_neg] = float("-inf")

        # only consider logits which are from a token in masked region
        if apply_cpc_over_masked_regions:
            logits = logits[mask_idx[bs], :]

        # store the logits for later aggregation
        logits_per_batch.append(logits)

    # create a single batch of predictions
    all_logits = torch.cat(logits_per_batch, dim=0) / temperature

    # the first class (positive) should be high (1 with cosine similarity),
    # all other classes (negative) should be low (-1 with cosine similarity)
    # therefore we can simply use cross entropy with target_class=0
    target_class = torch.zeros((all_logits.shape[0],), dtype=torch.long, device=device)
    loss = F.cross_entropy(all_logits, target_class, reduction=ce_reduction)

    return loss, (all_logits, target_class)


def diversity_loss(
    logits: torch.Tensor,
    sample_lengths: List[int],
    eps: float = 1e-7,
    scale_down: bool = False,
):
    # shape is [batch_size, sequence_length, num_codebooks, num_entries]
    batch_size, seq_length, num_codebooks, num_entries = logits.shape
    device = logits.device

    # diversity of the probabilities over the quantization vectors is maximized during
    # self-supervised training; this implies you want the logits to be as uniform as
    # possible
    probs = F.softmax(logits.to(torch.float32), dim=-1)

    # we want to ignore the softmax distributions from the time steps which are padded
    # as they are very common but not meaningful
    valid_idx = []
    for batch_idx, length in enumerate(sample_lengths):
        start_idx = batch_idx * seq_length
        end_idx = start_idx + length

        valid_idx.append(torch.arange(start_idx, end_idx, device=device))

    valid_idx = torch.cat(valid_idx)

    # reshape the probabilities so that we can easily compute entropy over each codebook
    # while also ignoring padded predictions
    valid_probs = probs.view(batch_size * seq_length, num_codebooks, num_entries)
    valid_probs = valid_probs[valid_idx, :]

    # mean prob for each codebook entry approximates the probability distribution
    approx_prob_dist = valid_probs.mean(dim=0)

    # based on the prob dist we can compute entropy and perplexity for each codebook
    entropy = -torch.sum(approx_prob_dist * torch.log(approx_prob_dist + eps), dim=-1)
    perplexity = torch.exp(entropy)

    # in the best case (uniform probability distribution) the perplexity of a codebook
    # is equal the number of entries, in the worst case (one-hot) the perplexity is 1.
    # Therefore, we subtract the actual perplexity from the maximum perplexity
    # such that we can get a minimizable loss
    loss_per_codebook = num_entries - perplexity

    # optionally scale loss so range is between 0 and 1
    # (0=uniform, 1=a single high prob)
    if scale_down:
        loss_per_codebook /= num_entries - 1
        loss = torch.mean(loss_per_codebook)
    else:
        loss = torch.sum(loss_per_codebook)

    return loss


def l2_norm_loss(x: torch.Tensor):
    # we use a l2 norm on the speech features to make sure they stay small-valued
    x = x.to(torch.float32)
    x = torch.pow(x, 2)
    x = torch.mean(x)

    return x


def codebook_similarity_loss(
    quantization_choices: torch.Tensor,
    num_codebooks: int,
    num_entries: int,
    std: bool = False,
):
    # first reshape into a batch of n codebooks
    quantization_choices = quantization_choices.view(num_codebooks, num_entries, -1)

    # norm each codebook vector to compute upper-triangular cosine distances
    codebook_normed = quantization_choices / torch.norm(
        quantization_choices, dim=2, keepdim=True
    )

    minimization_values = []
    for i in range(num_codebooks):
        # cosine distance matrix of codebook i
        codebook = codebook_normed[i]
        distance_matrix = torch.mm(codebook, codebook.transpose(0, 1))

        # get the upper triangular of the distance matrix, excluding the diagonal
        upper_tria_idx = torch.triu_indices(
            distance_matrix.shape[0], distance_matrix.shape[1], offset=1
        )
        upper_tria = distance_matrix[upper_tria_idx[0], upper_tria_idx[1]]

        # maxime the std so that there is as much variation as possible
        if std:
            minimization_values.append(-upper_tria.std())
        else:
            minimization_values.append(upper_tria.pow(2))

    # sum the std of each codebook and turn into maximisation by taking the negative
    to_minimize = torch.stack(minimization_values).sum()

    # take exp to make it monotonically decreasing to 0
    if std:
        return torch.exp(to_minimize)
    else:
        return to_minimize


########################################################################################
# network implementation


@dataclass
class ForSSLConfig:
    # contrastive loss settings
    num_dim_similarity: int = 256
    num_negative_samples: int = 100
    contrastive_temperature: float = 0.1
    ce_reduction: str = "sum"
    apply_cpc_over_masked_regions_only: bool = True
    sample_neg_from_masked_regions_only: bool = True

    # other loss settings (set to <= 0 to disable)
    diversity_loss_weight: float = 0.1
    diversity_loss_scale_down: bool = False
    diversity_loss_epsilon: float = 1e-7

    # extra regulation
    l2_loss_weight: float = 10

    # quantization settings
    num_codebooks: int = 2
    num_entries: int = 320
    num_dim_quantized: int = (
        256  # each codebook entry will have dim=num_dim_quantized/num codebooks
    )
    gumbel_temperature_start: float = 2
    gumbel_temperature_floor: float = 0.5
    gumbel_temperature_factor: float = 0.999995
    freeze_codebooks: bool = False
    freeze_codebook_linear: bool = False
    n_chunked_init: int = 0  # disabled

    # grad multiplication of CNN
    cnn_grad_factor: float = 0.1


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class SSLForwardResult(NamedTuple):
    # loss
    loss: torch.Tensor
    loss_contrastive: torch.Tensor
    loss_diversity: torch.Tensor
    loss_l2: torch.Tensor
    # metrics important for training progress
    cpc_logits: torch.Tensor
    cpc_targets: torch.Tensor
    codebook_logits: torch.Tensor
    # actual features
    speech_features: torch.Tensor
    context_features: torch.Tensor
    quantized_features: torch.Tensor
    # masks
    mask_idx: List[torch.Tensor]
    sample_lengths: List[int]


class Wav2vec2ForSSL(torch.nn.Module):
    def __init__(self, w2v2_cfg: Wav2vec2Config, ssl_cfg: ForSSLConfig):
        super().__init__()

        # w2v2 config is stored in w2v2 object
        self.ssl_cfg = ssl_cfg

        # backbone model
        self.w2v2 = Wav2vec2(w2v2_cfg)

        # quantization module - only used during self-supervised pre-training
        self.quantization_layer = QuantizationLayer(
            w2v2_cfg.num_dim_speech,
            ssl_cfg.num_codebooks,
            ssl_cfg.num_entries,
            ssl_cfg.num_dim_quantized,
            ssl_cfg.gumbel_temperature_start,
            ssl_cfg.n_chunked_init,
        )

        if ssl_cfg.freeze_codebooks:
            self.quantization_layer.quantization_choices.requires_grad_(False)
        if ssl_cfg.freeze_codebook_linear:
            self.quantization_layer.classification_layer.requires_grad_(False)

        # projects quantized and context features to the same dimension, so it's
        # possible to compute cosine similarity.
        self.project_context_feature = nn.Linear(
            w2v2_cfg.num_dim_context, ssl_cfg.num_dim_similarity
        )
        self.project_quantized_feature = nn.Linear(
            ssl_cfg.num_dim_quantized, ssl_cfg.num_dim_similarity
        )

    def quantize_features(self, speech_features: torch.Tensor):
        # input has shape [BATCH_SIZE, SEQ_LENGTH, NUM_SPEECH_FEATURES]
        return self.quantization_layer(speech_features)

    def step_gumbel_temperature(self):
        self.quantization_layer.temp = nn.Parameter(
            torch.clip(
                self.quantization_layer.temp * self.ssl_cfg.gumbel_temperature_factor,
                min=self.ssl_cfg.gumbel_temperature_floor,
            ),
            requires_grad=False,
        )

    def get_gumbel_temperature(self):
        return self.quantization_layer.temp.item()

    def get_codebooks(self):
        return (
            self.quantization_layer.quantization_choices,
            self.quantization_layer.num_codebooks,
            self.quantization_layer.num_entries,
        )

    def forward(
        self, raw_audio: torch.Tensor, sample_lengths: List[int]
    ) -> SSLForwardResult:
        # speech features from raw audio
        speech_features, sample_lengths = self.w2v2.speech_features(
            raw_audio, sample_lengths
        )

        if self.ssl_cfg.cnn_grad_factor != 1:
            speech_features = GradMultiply.apply(
                speech_features, self.ssl_cfg.cnn_grad_factor
            )

        # quantize the speech features
        quantized_features, codebook_logits = self.quantize_features(speech_features)

        # context features with transformer
        context_features, masked_idx = self.w2v2.context_features(
            speech_features, sample_lengths, require_mask=True
        )

        # project context and quantized features to same dimension
        context_features = self.project_context_feature(context_features)
        quantized_features = self.project_quantized_feature(quantized_features)

        # compute contrastive loss (cpc=contrastive-predictive coding)
        loss_contrastive, (cpc_logits, cpc_targets) = contrastive_loss(
            quantized_features,
            context_features,
            masked_idx,
            sample_lengths,
            self.ssl_cfg.num_negative_samples,
            self.ssl_cfg.contrastive_temperature,
            self.ssl_cfg.ce_reduction,
            self.ssl_cfg.apply_cpc_over_masked_regions_only,
            self.ssl_cfg.sample_neg_from_masked_regions_only,
        )

        # compute diversity loss
        if self.ssl_cfg.diversity_loss_weight > 0:
            loss_diversity = diversity_loss(
                codebook_logits,
                sample_lengths,
                self.ssl_cfg.diversity_loss_epsilon,
                self.ssl_cfg.diversity_loss_scale_down,
            )
            loss_diversity *= self.ssl_cfg.diversity_loss_weight
        else:
            loss_diversity = 0

        # compute l2 norm loss on speech features
        if self.ssl_cfg.l2_loss_weight > 0:
            loss_l2 = l2_norm_loss(speech_features)
            loss_l2 *= self.ssl_cfg.l2_loss_weight
        else:
            loss_l2 = 0

        loss = loss_contrastive + loss_diversity + loss_l2

        return SSLForwardResult(
            loss,
            loss_contrastive,
            loss_diversity,
            loss_l2,
            cpc_logits,
            cpc_targets,
            codebook_logits,
            speech_features,
            context_features,
            quantized_features,
            masked_idx,
            sample_lengths,
        )
