import torch
from rdr import RDRTrainer, Divergence


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))

    def forward(self, x):
        return torch.sigmoid(self.net(x)) * 2.0


def test_trainer_runs_and_returns_history():
    x_real = torch.randn(32, 4)
    x_gen = torch.randn(32, 4)

    trainer = RDRTrainer(model=TinyNet(), divergence=Divergence.HELLINGER)
    model, history = trainer.fit(x_real, x_gen, num_epochs=3, batch_size=16, verbose=False)

    assert model is not None
    assert len(history) == 3


def test_trainer_exposes_balanced_mixture_ratio():
    trainer = RDRTrainer(model=TinyNet(), divergence=Divergence.HELLINGER)
    assert trainer.mixture_ratio == 0.5


def test_scores_are_bounded_in_zero_to_two():
    x_real = torch.randn(16, 4)
    x_gen = torch.randn(16, 4)

    trainer = RDRTrainer(model=TinyNet(), divergence=Divergence.HELLINGER)
    trainer.fit(x_real, x_gen, num_epochs=2, batch_size=8, verbose=False)
    scores = trainer.score(x_real)

    assert torch.all(scores >= 0.0)
    assert torch.all(scores <= 2.0)


def test_score_does_not_apply_a_second_sigmoid():
    model = TinyNet()
    x = torch.randn(16, 4)
    trainer = RDRTrainer(model=model, divergence=Divergence.HELLINGER, device="cpu")

    with torch.no_grad():
        expected = model(x).reshape(-1)

    assert torch.equal(trainer.score(x), expected)


def test_trainer_accepts_different_sample_counts():
    x_real = torch.randn(23, 4)
    x_gen = torch.randn(31, 4)
    trainer = RDRTrainer(model=TinyNet(), divergence=Divergence.HELLINGER, device="cpu")

    _, history = trainer.fit(x_real, x_gen, num_epochs=1, batch_size=8, verbose=False)

    assert len(history) == 1


def test_equal_distributions_have_unit_ratio_as_population_optimum():
    class ConstantRatio(torch.nn.Module):
        def __init__(self, value: float):
            super().__init__()
            self.logit = torch.nn.Parameter(torch.logit(torch.tensor(value / 2.0)))

        def forward(self, x):
            return (2.0 * torch.sigmoid(self.logit)).expand(x.size(0), 1)

    x = torch.randn(64, 4)
    unit_loss = RDRTrainer(ConstantRatio(1.0), device="cpu")._loss(x, x)
    off_target_loss = RDRTrainer(ConstantRatio(1.5), device="cpu")._loss(x, x)

    assert unit_loss < off_target_loss


def test_automatic_splits_track_validation_and_test_loss():
    trainer = RDRTrainer(TinyNet(), device="cpu")
    _, history = trainer.fit(
        torch.randn(40, 4),
        torch.randn(50, 4),
        num_epochs=3,
        batch_size=8,
        validation_fraction=0.2,
        test_fraction=0.1,
        split_seed=17,
        verbose=False,
    )

    assert len(trainer.validation_history) == len(history)
    assert trainer.best_epoch is not None
    assert trainer.test_loss is not None


def test_validation_loss_drives_early_stopping():
    trainer = RDRTrainer(TinyNet(), lr=0.0, weight_decay=0.0, device="cpu")
    _, history = trainer.fit(
        torch.randn(40, 4),
        torch.randn(40, 4),
        num_epochs=20,
        batch_size=8,
        validation_fraction=0.2,
        early_stopping_patience=2,
        min_delta=1e-5,
        verbose=False,
    )

    assert len(history) == 3
    assert trainer.stopped_epoch == 2


def test_all_divergences_are_available():
    assert Divergence.HELLINGER.value == "hellinger"
    assert Divergence.KL.value == "kl"
    assert Divergence.CHISQ.value == "chisq"
