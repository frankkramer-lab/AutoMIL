# Trainer

`automil.trainer.Trainer` orchestrates training MIL models using Slideflow and FastAI.
It provides automatic batch size adjustment based on GPU memory constraints, optional early stopping
and k-fold cross-validation with the ability of providing additional callback.

::: automil.trainer.Trainer