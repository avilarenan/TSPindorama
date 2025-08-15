# Wrapper

A TSPindorama-ready, pluggable implementation of CATS (Auxiliary Time Series)
that can wrap *any* predictor in your framework.

## Features:
- Multiple ATS constructor types: 'linear', 'conv', 'attention' (lightweight)
- Enforces paper-inspired constraints: continuity (conv smoothing), sparsity (L1),
  variability/diversity loss (decorrelation among ATS heads)
- Generic CATSWrapper that concatenates ATS to the encoder input, forwards
  to the wrapped model, and projects the model's full output back to the
  original OTS dimensionality (so ATS are treated as context and not part
  of the final loss).
- Returns predictions and a dictionary of regularization terms so your
  training loop can add the proper penalties (L1, diversity, continuity).

## Usage (example):
    
  ```python
  from models.cats_wrapper import CATSWrapper, ATSConstructor
  # model: any model that accepts x_enc as first positional arg with shape
  # (batch, seq_len, n_features) and returns (batch, pred_len, n_out_features)
  base_model = YourModel(input_size = orig_dim + ats_dim, ...)
  cats = CATSWrapper(base_model, orig_dim=orig_dim, ats_dim=4, constructor='conv')

  # forward:
  pred, regs = cats(x_enc, *other_args, **other_kwargs)
  # compute loss only on pred (shape must be (batch, pred_len, orig_dim))
  loss = mse(pred, y_true) + lambda1 * regs['l1'] + lambda2 * regs['diversity']
  ```

## Notes on model configuration:
- For best compatibility, instantiate your underlying model so that its
  input/output channels correspond to (orig_dim + ats_dim). That way the
  model will learn to produce a full multivariate forecast which the wrapper
  will project back to orig_dim.
- The wrapper also handles the case where the wrapped model already returns
  predictions with dimension == orig_dim (no projection applied).

# References:

Implementation inspired by:

- CATS Paper: https://arxiv.org/abs/2403.01673

- CATS github: https://github.com/LJC-FVNR/CATS