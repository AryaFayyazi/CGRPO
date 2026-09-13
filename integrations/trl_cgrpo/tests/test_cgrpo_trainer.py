import random

import numpy as np
import pytest
import torch
from datasets import Dataset, load_dataset
from trl_cgrpo import CGRPOConfig, CGRPOTrainer
from trl_cgrpo.conformal import (
    aps_score,
    conformal_quantile,
    first_success_score,
    pass_rate_score,
    prediction_set,
    select_delta_auto,
)


class TrlTestCase:
    @pytest.fixture(autouse=True)
    def set_tmp_dir(self, tmp_path):
        self.tmp_dir = str(tmp_path)


class TestConformal:
    def test_quantile_uses_finite_sample_index(self):
        scores = [0.1 * i for i in range(1, 11)]  # n = 10
        # ceil(11 * 0.8) = 9 -> 9th smallest
        assert conformal_quantile(scores, 0.2) == pytest.approx(0.9)
        # index overruns n -> largest score
        assert conformal_quantile(scores, 0.01) == pytest.approx(1.0)
        assert conformal_quantile([], 0.1) == 1.0

    def test_quantile_marginal_coverage(self):
        rng = np.random.default_rng(0)
        n, delta, trials = 50, 0.2, 4000
        hits = 0
        for _ in range(trials):
            s = rng.uniform(size=n + 1)
            hits += s[n] <= conformal_quantile(s[:n], delta)
        coverage = hits / trials
        # Theorem: 1 - delta <= coverage <= 1 - delta + 1/(n+1); allow Monte Carlo error.
        assert 1 - delta - 0.02 <= coverage <= 1 - delta + 1 / (n + 1) + 0.02

    def test_aps_score(self):
        rng = random.Random(0)
        assert aps_score("7", ["1", "2", "3"], 3, rng) == 1.0  # gold never sampled
        assert 0.0 <= aps_score("7", ["7", "7", "7", "1"], 4, rng) <= 0.75  # gold is the mode
        assert aps_score("1", ["7", "7", "7", "1"], 4, rng) >= 0.75  # mode's mass comes first

    def test_prediction_set(self):
        rng = random.Random(0)
        assert prediction_set(["7", "7", "7", "1"], 4, 0.5, rng) == ["7"]
        assert set(prediction_set(["7", "7", "1", "1"], 4, 0.9, rng)) == {"7", "1"}
        assert prediction_set(["", ""], 2, 0.5, rng) == []  # failed extractions carry no mass

    def test_execution_scores(self):
        assert first_success_score([False, True, True, False], 4) == 0.5
        assert first_success_score([False] * 4, 4) == 1.0
        assert pass_rate_score([False, True, True, False], 4) == 0.5

    def test_select_delta_auto(self):
        delta, solve_rate = select_delta_auto([0.2, 1.0, 0.4, 1.0])
        assert solve_rate == 0.5
        assert delta == pytest.approx(0.55)
        assert select_delta_auto([0.1] * 10)[0] == 0.05  # clipped


class TestCGRPOConfig:
    def test_num_generations_defaults_to_largest_budget(self):
        args = CGRPOConfig("dummy", budget_grid=[4, 2, 8], per_device_train_batch_size=8)
        assert args.budget_grid == [2, 4, 8]
        assert args.num_generations == 8

    def test_num_generations_must_match_largest_budget(self):
        with pytest.raises(ValueError, match="must equal max"):
            CGRPOConfig("dummy", budget_grid=[2, 4], num_generations=8, per_device_train_batch_size=8)

    @pytest.mark.parametrize("grid", [[], [0, 2], [2, 2]])
    def test_invalid_budget_grid(self, grid):
        with pytest.raises(ValueError, match="budget_grid"):
            CGRPOConfig("dummy", budget_grid=grid)

    def test_invalid_score(self):
        with pytest.raises(ValueError, match="score"):
            CGRPOConfig("dummy", budget_grid=[2], score="nope", per_device_train_batch_size=2)


def length_reward(completions, **kwargs):
    return [float(len(c)) for c in completions]


class RecordingCGRPOTrainer(CGRPOTrainer):
    """Keeps every generation batch as returned, before GRPOTrainer shuffles and splits it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorded = []

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        self.recorded.append({k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in output.items()})
        return output


class TestCGRPOTrainer(TrlTestCase):
    def _setup(self, **config_kwargs):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        calibration_dataset = Dataset.from_dict({"prompt": dataset["prompt"][:4], "answer": ["x"] * 4})
        args = CGRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=4,
            budget_grid=[2, 4],
            max_completion_length=8,
            max_steps=2,
            recalibrate_every=1,
            report_to="none",
            **config_kwargs,
        )
        return dataset, calibration_dataset, args

    def test_early_stopping_pads_and_masks(self):
        dataset, calibration_dataset, args = self._setup()
        trainer = RecordingCGRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=length_reward,
            args=args,
            train_dataset=dataset,
            calibration_dataset=calibration_dataset,
            answer_extractor=lambda text: "x",  # every group agrees -> singleton set at the first budget
        )
        previous = {n: p.clone() for n, p in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert set(trainer.qhats) == {2, 4} and trainer.delta is not None
        assert len(trainer.recorded) == 2
        for batch in trainer.recorded:
            # one group of width 4 that stopped at k=2: rows 2 and 3 are padding
            assert batch["completion_mask"][2:].sum() == 0
            assert torch.all(batch["advantages"][2:] == 0)
            assert batch["completion_mask"][:2].sum() > 0
            assert batch["num_items_in_batch"] == batch["completion_mask"].sum()
        assert any(not torch.equal(p, previous[n]) for n, p in trainer.model.named_parameters())

    def test_unresolved_prompts_use_full_budget(self):
        dataset, calibration_dataset, args = self._setup()
        trainer = RecordingCGRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=length_reward,
            args=args,
            train_dataset=dataset,
            calibration_dataset=calibration_dataset,
            answer_extractor=lambda text: "",  # extraction always fails -> set never becomes a singleton
        )
        trainer.train()
        assert trainer.qhats == {2: 1.0, 4: 1.0}
        for batch in trainer.recorded:
            assert torch.all(batch["completion_mask"].sum(dim=1) > 0)  # nothing padded

    def test_first_success_score(self):
        dataset, calibration_dataset, args = self._setup(score="first_success")

        def always_passes(completions, **kwargs):
            return [True] * len(completions)

        trainer = RecordingCGRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=length_reward,
            args=args,
            train_dataset=dataset,
            calibration_dataset=calibration_dataset,
            verifier=always_passes,
        )
        trainer.train()
        # every calibration example passes at position 1, so qhat_k = 1/k and a first-position pass resolves at k=2
        assert trainer.qhats == {2: pytest.approx(0.5), 4: pytest.approx(0.25)}
        for batch in trainer.recorded:
            assert batch["completion_mask"][2:].sum() == 0

    def test_requires_calibration_dataset(self):
        dataset, _, args = self._setup()
        with pytest.raises(ValueError, match="calibration_dataset"):
            CGRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs=length_reward,
                args=args,
                train_dataset=dataset,
                answer_extractor=lambda text: "x",
            )

    def test_aps_requires_answer_extractor(self):
        dataset, calibration_dataset, args = self._setup()
        with pytest.raises(ValueError, match="answer_extractor"):
            CGRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs=length_reward,
                args=args,
                train_dataset=dataset,
                calibration_dataset=calibration_dataset,
            )
