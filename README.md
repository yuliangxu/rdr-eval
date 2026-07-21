# RDR Python Package

This package provides a lightweight PyTorch interface for relative density ratio (RDR) estimation and generative-model comparison. It is based on the paper
*Distributional Evaluation of Generative Models via Relative Density Ratio* by Yuliang Xu, Yun Wei, and Li Ma. <https://arxiv.org/abs/2510.25507>

**Documentation:** <https://yuliangxu.github.io/rdr-eval/> ·
**Source:** <https://github.com/yuliangxu/rdr-eval>

For a real-data distribution $p$ and generated-data distribution $q$, the package estimates

$$
r(x) = \frac{2p(x)}{p(x)+q(x)} \in (0,2).
$$

Values above 1 indicate that $x$ is more characteristic of the real distribution; values below 1 indicate that it is more characteristic of the generated distribution. If $p=q$, the population ratio is 1 everywhere.

## What is included

- A training API for estimating density ratios between real and generated samples
- Support for three $\phi$-divergences:
  - squared Hellinger
  - KL
  - chi-squared
- A model-agnostic interface for user-supplied PyTorch networks

## Installation

From PyPI:

```bash
python3 -m pip install rdr-eval
```

For local development from the package root:

```bash
python3 -m pip install -e .
```

## Quick start

```python
import torch
from rdr import RDRTrainer, Divergence

x_real = torch.randn(256, 8)
x_gen = torch.randn(256, 8)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        # This is the one and only output activation. RDRTrainer does not
        # apply another sigmoid.
        return 2.0 * torch.sigmoid(self.net(x))

trainer = RDRTrainer(model=Net(), divergence=Divergence.HELLINGER)
model, history = trainer.fit(x_real, x_gen, num_epochs=50)
ratio = trainer.score(x_real[:10])
```

### Validation, early stopping, and testing

The trainer can make reproducible splits independently within the real and generated samples:

```python
model, train_loss = trainer.fit(
    x_real,
    x_gen,
    num_epochs=500,
    validation_fraction=0.2,
    test_fraction=0.1,
    split_seed=123,
    early_stopping_patience=20,
    early_stopping_start=50,
    min_delta=1e-5,
    restore_best=True,
)

validation_loss = trainer.validation_history
best_epoch = trainer.best_epoch
test_loss = trainer.test_loss
```

Validation loss controls early stopping and optional learning-rate scheduling. When `restore_best=True`, the model is restored to the lowest-validation-loss state. The test split is evaluated only once, after model selection. For externally managed splits, pass `x_real_val`, `x_gen_val`, `x_real_test`, and `x_gen_test` instead of the fraction arguments.

## Output activation contract

The model passed to `RDRTrainer` must return the final density-ratio estimate, rather than an untransformed logit. For the standard RDR range `(0, 2)`, end the model with

```python
return 2.0 * torch.sigmoid(logits)
```

Do not apply a second sigmoid when calling `score()` or `evaluate_ratio()`; both functions return the model output directly. Other positive-output parameterizations can be used when appropriate for a particular objective, but the bounded sigmoid is the default recommended choice for RDR.

## One-dimensional simulation

The example in [`examples/gaussian_1d.py`](examples/gaussian_1d.py) uses

$$p=N(0,1), \qquad q=N(1,1),$$

for which the exact RDR is available analytically. Run it with:

```bash
python3 examples/gaussian_1d.py
```

It trains a small network, evaluates it on a grid, and prints the estimated ratio beside the analytical ratio and the grid mean absolute error. No plotting library is required.

For illustrated, notebook-style walkthroughs based on the toy experiments in the original RDR repository, install the example dependency and run:

```bash
python3 -m pip install -e ".[examples]"
python3 examples/plot_rdr_1d.py
python3 examples/plot_rdr_2d.py
```

The files use `# %%` cells and Sphinx-Gallery narrative sections, so they can also be opened interactively as notebooks in editors that support Python cells.

Rendered walkthroughs with figures and downloadable notebooks are available in the [example gallery](https://yuliangxu.github.io/rdr-eval/auto_examples/index.html).

## Choosing a divergence

```python
trainer = RDRTrainer(model, divergence=Divergence.HELLINGER)  # default
trainer = RDRTrainer(model, divergence=Divergence.KL)
trainer = RDRTrainer(model, divergence=Divergence.CHISQ)
```

All objectives form their denominator expectation under

$$m(x)=\xi p(x)+(1-\xi)q(x),$$

where `mixture_ratio` is $\xi$. Its default value `0.5` therefore estimates $p/m=2p/(p+q)$. Real and generated tensors may contain different numbers of samples, but their remaining dimensions must match.

## Notes

- Inputs are converted to `torch.float32` and moved to the trainer's device.
- The supplied network should return one scalar per observation.
- Call `trainer.score(x)` for the final RDR estimate.

## Development and release checks

Install the development tools, run the tests, and validate both distribution formats:

```bash
python3 -m pip install -e ".[dev,examples]"
python3 -m pytest
python3 -m build
python3 -m twine check dist/*
```

Upload to TestPyPI before publishing a release to PyPI:

```bash
python3 -m twine upload --repository testpypi dist/*
```

## License

RDR Eval is released under the [MIT License](LICENSE).
