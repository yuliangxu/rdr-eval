"""Estimate an RDR whose population value is known analytically.

Run from the project root with:
    python3 examples/gaussian_1d.py
"""

import torch

from rdr import Divergence, RDRTrainer


class RatioNet(torch.nn.Module):
    """Small MLP with exactly one bounded output activation."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.layers(x)
        return 2.0 * torch.sigmoid(logits)


def exact_rdr(x: torch.Tensor, generated_mean: float) -> torch.Tensor:
    """Return 2p/(p+q) for p=N(0,1), q=N(generated_mean,1)."""

    # log(q(x) / p(x)) for equal-variance unit Gaussians.
    log_q_over_p = generated_mean * x - 0.5 * generated_mean**2
    return 2.0 * torch.sigmoid(-log_q_over_p)


def main() -> None:
    torch.manual_seed(7)
    generated_mean = 1.0

    x_real = torch.randn(2_000, 1)
    x_generated = torch.randn(1_600, 1) + generated_mean

    trainer = RDRTrainer(
        model=RatioNet(),
        divergence=Divergence.HELLINGER,
        lr=1e-3,
        weight_decay=0.0,
        device="cpu",
    )
    _, history = trainer.fit(
        x_real,
        x_generated,
        num_epochs=200,
        batch_size=256,
        verbose=False,
    )

    grid = torch.linspace(-2.0, 3.0, 11).unsqueeze(1)
    estimated = trainer.score(grid).cpu()
    target = exact_rdr(grid, generated_mean).reshape(-1)
    mae = torch.mean(torch.abs(estimated - target))

    print(f"final training loss: {history[-1]:.6f}")
    print(f"grid mean absolute error: {mae.item():.6f}")
    print("\n    x    estimated    exact")
    for x_value, estimate, truth in zip(grid.reshape(-1), estimated, target):
        print(f"{x_value.item():5.1f}    {estimate.item():8.4f}  {truth.item():8.4f}")


if __name__ == "__main__":
    main()
