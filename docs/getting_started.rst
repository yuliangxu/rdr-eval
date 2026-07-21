Getting started
===============

Installation
------------

Install the package from PyPI:

.. code-block:: bash

   python -m pip install rdr-eval

The distribution is named ``rdr-eval`` and the import package is named
``rdr``.

Basic usage
-----------

The supplied model owns the single output activation and must return the final
ratio estimate in ``(0, 2)``:

.. code-block:: python

   import torch
   from rdr import Divergence, RDRTrainer

   class RatioNet(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self.layers = torch.nn.Sequential(
               torch.nn.Linear(8, 64),
               torch.nn.ReLU(),
               torch.nn.Linear(64, 1),
           )

       def forward(self, x):
           return 2.0 * torch.sigmoid(self.layers(x))

   x_real = torch.randn(1_000, 8)
   x_generated = torch.randn(1_000, 8) + 0.25

   trainer = RDRTrainer(RatioNet(), divergence=Divergence.HELLINGER)
   model, train_loss = trainer.fit(
       x_real,
       x_generated,
       validation_fraction=0.2,
       test_fraction=0.1,
       early_stopping_patience=20,
       restore_best=True,
   )

   ratios = trainer.score(x_real[:10])

Validation and testing
----------------------

Validation loss controls early stopping and optional learning-rate scheduling.
The test split is evaluated only after model selection. Results are available
as ``trainer.validation_history``, ``trainer.best_epoch``,
``trainer.stopped_epoch``, and ``trainer.test_loss``.
