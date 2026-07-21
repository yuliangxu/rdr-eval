import torch
from rdr import RDRTrainer, Divergence


class SimpleRatioNet(torch.nn.Module):
    def __init__(self, input_dim: int = 8):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        # The model returns the final ratio; the trainer does not reactivate it.
        return 2.0 * torch.sigmoid(self.net(x))


if __name__ == "__main__":
    torch.manual_seed(0)
    x_real = torch.randn(256, 8)
    x_gen = torch.randn(256, 8)

    trainer = RDRTrainer(model=SimpleRatioNet(8), divergence=Divergence.KL, device="cpu")
    model, history = trainer.fit(x_real, x_gen, num_epochs=20, batch_size=64, verbose=True)
    scores = trainer.score(x_real)
    print("score count:", scores.shape[0])
    print("score range:", float(scores.min()), float(scores.max()))
