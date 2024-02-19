import torch
import torch as t
import pytest

from nanow2v2.model.wav2vec2_ssl import diversity_loss, contrastive_loss, l2_norm_loss


class TestDiversityLoss:
    def test_bs1(self):
        # batch size 1, sequence length 3, 1 codebook, 2 entries = 6
        logits = t.tensor([[1, 1], [1, 2], [3, 1]], dtype=t.float32).reshape(
            (1, 3, 1, 2)
        )

        r = diversity_loss(logits, [3], eps=0)
        r = r.item()

        # manual calculation: answer should be 0.001
        assert r == pytest.approx(0.001, abs=1e-2)

    def test_uniform(self):
        logits = torch.ones((2, 3, 1, 2), dtype=t.float32)
        r = diversity_loss(logits, [3, 3], eps=0).item()

        assert r == pytest.approx(0)

    def test_spike(self):
        logits = t.tensor([[1000, 1], [1000, 1], [1000, 1]], dtype=t.float32).reshape(
            (1, 3, 1, 2)
        )
        r = diversity_loss(logits, [3], eps=1e-6).item()

        assert r == pytest.approx(1)

    def test_padded_sequence(self):
        # create a batch such that loss should be 0. We add one-hot probabilities for
        # padded indexes
        batch = []
        sample_length = []

        bs = 4
        seq_length = 10
        num_dim = 5
        for i in range(bs):
            num_padded = i
            normal = torch.ones((seq_length - num_padded, num_dim)) / num_dim
            padded = torch.eye(num_dim)[0:1, :].expand(num_padded, -1)
            sample_length.append(seq_length - num_padded)

            seq = torch.cat([normal, padded], dim=0)
            batch.append(seq)

        logits = torch.stack(batch)[:, :, None, :]
        r = diversity_loss(logits, sample_length, eps=0).item()
        assert r == pytest.approx(0)

        # if we include padded regions loss should be higher
        r_pad = diversity_loss(logits, [seq_length for _ in range(bs)], eps=0).item()
        assert r_pad > r
        assert r_pad != pytest.approx(0)


class TestContrastiveLoss:
    def test_bs1(self):
        # we create a fake batch of shape [1, 9, 2]
        quantized_features = torch.arange(18).reshape((1, 9, 2)).to(torch.float32)
        context_features = torch.arange(18).reshape((1, 9, 2)).to(torch.float32)
        context_features = context_features[:, [i for i in reversed(range(9))], :]
        sample_lengths = [9]

        neg_idx = [torch.tensor([[3, 4, 5]]).expand(9, -1)]

        loss, _ = contrastive_loss(
            quantized_features,
            context_features,
            neg_idx,
            sample_lengths,
            temperature=1,
            ce_reduction="none",
        )

        # manually computed
        assert loss[0].item() == pytest.approx(1.5963, abs=1e-3)
        assert loss[3].item() == pytest.approx(1.0988, abs=1e-3)

    def test_correct_is_lower(self):
        quantized_features = torch.arange(18).reshape((1, 9, 2)).to(torch.float32)
        context_features = torch.arange(18).reshape((1, 9, 2)).to(torch.float32)
        reversed_context_features = context_features[
            :, [i for i in reversed(range(9))], :
        ]

        sample_lengths = [9]
        neg_idx = [torch.tensor([[3, 4, 5]]).expand(9, -1)]

        loss_equal, _ = contrastive_loss(
            quantized_features,
            context_features,
            neg_idx,
            sample_lengths,
            temperature=0.1,
            ce_reduction="sum",
        )
        loss_reversed, _ = contrastive_loss(
            quantized_features,
            reversed_context_features,
            neg_idx,
            sample_lengths,
            temperature=0.1,
            ce_reduction="sum",
        )

        assert loss_reversed > loss_equal


class TestL2NormLoss:
    def test_bs1(self):
        values = t.tensor([-1, 1, -4, 4, 1, 1])
        r = l2_norm_loss(values).item()

        assert r == pytest.approx(6)

    def test_zero(self):
        values = t.zeros((300, 500, 100))
        r = l2_norm_loss(values).item()

        assert r == pytest.approx(0)

    def test_decrease_increase(self):
        values = t.rand((10, 20, 30))
        values_lower = values / 10
        values_higher = values * 10

        r = l2_norm_loss(values).item()
        r_lower = l2_norm_loss(values_lower).item()
        r_higher = l2_norm_loss(values_higher).item()

        assert r_lower < r < r_higher
