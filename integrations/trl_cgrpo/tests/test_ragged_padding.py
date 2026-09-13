"""
The one assumption the TRL integration rests on: that NaN-padding a ragged
group leaves TRL's advantage computation correct.

TRL computes the group baseline with `rewards.view(-1, num_generations)` and
nan-aware reductions. C-GRPO pads each group out to K_max, so these tests pin
down that padded slots are excluded from the baseline and receive zero
advantage. If TRL ever changes to a non-nan-aware reduction, these fail and the
integration needs revisiting.

Requires trl; skipped when it is not installed.
"""
import pytest

torch = pytest.importorskip("torch")
nanstd = pytest.importorskip("trl.trainer.utils").nanstd

G = 4  # K_max


def _advantages(rewards):
    """Reproduce TRL's group-baseline advantage computation."""
    r = torch.tensor(rewards)
    pad = torch.isnan(r)
    mean = torch.nanmean(r.view(-1, G), dim=1).repeat_interleave(G, dim=0)
    adv = r - mean
    adv[pad] = 0.0
    return adv, mean, pad


def test_baseline_ignores_padded_slots():
    # prompt A stopped at k=2, prompt B ran the full k=4
    adv, mean, _ = _advantages([1.0, 0.0, float("nan"), float("nan"),
                                1.0, 1.0, 0.0, 1.0])
    assert mean[0].item() == pytest.approx(0.5)     # (1+0)/2, not /4
    assert mean[4].item() == pytest.approx(0.75)


def test_padded_slots_get_zero_advantage():
    adv, _, pad = _advantages([1.0, 0.0, float("nan"), float("nan"),
                               1.0, 1.0, 0.0, 1.0])
    assert torch.all(adv[pad] == 0.0)


def test_real_slots_keep_nonzero_advantage():
    adv, _, pad = _advantages([1.0, 0.0, float("nan"), float("nan"),
                               1.0, 1.0, 0.0, 1.0])
    assert adv[0].item() == pytest.approx(0.5)
    assert adv[1].item() == pytest.approx(-0.5)


def test_zero_variance_ragged_group_is_prunable():
    """An all-correct early-exit group must produce no gradient signal."""
    adv, _, _ = _advantages([1.0, 1.0, float("nan"), float("nan")] * 2)
    assert torch.nansum(adv.abs()).item() == pytest.approx(0.0)


def test_nanstd_ignores_padding_and_is_the_sample_std():
    """TRL's nanstd is ddof=1 over the NON-padded entries.

    Worth pinning: with scale_rewards="group" the advantage is divided by this,
    so whether padding counts toward the denominator changes the gradient scale.
    Here n=2 real entries, so the sample std is used, not the population std.
    """
    r = torch.tensor([1.0, 0.0, float("nan"), float("nan")])
    got = nanstd(r.view(-1, G), dim=1)[0].item()
    assert got == pytest.approx(
        torch.tensor([1.0, 0.0]).std(unbiased=True).item(), abs=1e-6)
    assert got != pytest.approx(
        torch.tensor([1.0, 0.0]).std(unbiased=False).item(), abs=1e-6)


def test_fully_padded_group_does_not_crash():
    """Defensive: a prompt that produced nothing must not poison the batch."""
    adv, _, pad = _advantages([float("nan")] * G + [1.0, 1.0, 0.0, 1.0])
    assert torch.all(adv[:G] == 0.0)
    assert torch.isfinite(adv[G:]).all()
