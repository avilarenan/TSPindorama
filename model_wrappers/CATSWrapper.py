
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# ATS Constructors
# --------------------------------------------------
class ATSConstructor(nn.Module):
    """Learned Auxiliary Time Series constructors.

    Supported types:
      - 'linear' : per-time-step linear mixing of input series -> K aux.
      - 'conv'   : temporal conv applied after a linear mixing (continuity).
      - 'attention': a light attention-style constructor (query from time -> mix series)

    All constructors produce output shape (batch, seq_len, ats_dim).
    """

    def __init__(self, orig_dim: int, ats_dim: int = 4, constructor: str = 'conv',
                 conv_kernel: int = 5, attn_heads: int = 2, hidden: int = 64):
        super().__init__()
        self.orig_dim = orig_dim
        self.ats_dim = ats_dim
        self.constructor = constructor

        if constructor not in ('linear', 'conv', 'attention'):
            raise ValueError("constructor must be one of 'linear','conv','attention'")

        # linear projection from orig_dim -> ats_dim (applied per time step)
        print(f"Using ATS constructor: {constructor} with orig_dim={orig_dim}, ats_dim={ats_dim}")
        self.proj = nn.Linear(orig_dim, ats_dim, bias=True)

        if constructor == 'conv':
            pad = (conv_kernel - 1) // 2
            # temporal conv expects (batch, channels, seq_len)
            self.temporal_conv = nn.Conv1d(ats_dim, ats_dim, kernel_size=conv_kernel, padding=pad, groups=1)
            # Optional small MLP for non-linearity
            self.post_mlp = nn.Sequential(nn.Linear(ats_dim, hidden), nn.ReLU(), nn.Linear(hidden, ats_dim))

        if constructor == 'attention':
            # light-weight attention: produce queries from time embeddings, keys/values from series
            # We'll implement a per-time self-attention across series dimension in a cheap way
            # Use linear layers to map series -> K/V and time -> Q
            self.q_net = nn.Linear(orig_dim, ats_dim)
            self.k_net = nn.Linear(orig_dim, ats_dim)
            self.v_net = nn.Linear(orig_dim, ats_dim)
            self.attn_out = nn.Linear(ats_dim, ats_dim)

        # per-channel gating (helps sparsity in practice)
        self.channel_gate = nn.Parameter(torch.ones(ats_dim))

        # initialize
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.proj.weight)
        if hasattr(self, 'temporal_conv'):
            nn.init.kaiming_normal_(self.temporal_conv.weight, nonlinearity='relu')
        if hasattr(self, 'q_net'):
            nn.init.xavier_uniform_(self.q_net.weight)
            nn.init.xavier_uniform_(self.k_net.weight)
            nn.init.xavier_uniform_(self.v_net.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, orig_dim)
        returns: ats (batch, seq_len, ats_dim)
        """
        b, t, d = x.shape
        # per-time projection
        x_flat = x.view(-1, d)  # (b*t, d)
        aux_flat = self.proj(x_flat)  # (b*t, ats_dim)
        aux = aux_flat.view(b, t, self.ats_dim)  # (b, t, k)

        if self.constructor == 'conv':
            # temporal conv expects (b, k, t)
            aux_t = aux.permute(0, 2, 1)
            aux_t = self.temporal_conv(aux_t)
            aux_t = aux_t.permute(0, 2, 1)
            aux = aux + self.post_mlp(aux_t)

        elif self.constructor == 'attention':
            # simple attention across series at each time: Q from time-aggregated vector
            # here we compute per-time queries by summing series -> shape (b, t, d)
            # but we already have x; implement scaled dot-product between time-wise q and k
            Q = self.q_net(x)  # (b, t, k)
            K = self.k_net(x)  # (b, t, k)
            V = self.v_net(x)  # (b, t, k)
            # compute attention weights across time (so continuity + global mixing)
            # Attention: for each batch and feature-dim compute softmax over time
            # We'll compute (b, k, t) matmul with (b, k, t) -> (b, t, t) then apply to V
            Q_t = Q.permute(0, 2, 1)  # (b, k, t)
            K_t = K.permute(0, 2, 1)  # (b, k, t)
            attn_logits = torch.matmul(Q_t.transpose(1, 2), K_t) / (self.ats_dim ** 0.5)  # (b, t, t)
            attn = torch.softmax(attn_logits, dim=-1)
            V_t = V.permute(0, 2, 1)  # (b, k, t)
            out_t = torch.matmul(attn, V_t.transpose(1, 2))  # (b, t, k)
            aux = self.attn_out(out_t)

        # gating
        gated = aux * torch.sigmoid(self.channel_gate)
        return gated


# --------------------------------------------------
# Wrapper
# --------------------------------------------------
class CATSWrapper(nn.Module):
    """Wrap any forecasting model to add CATS functionality.

    Parameters
    ----------
    model : nn.Module
        The forecasting model to be wrapped. It should accept the encoder
        input as the first positional argument with shape (batch, seq_len, n_features)
        and return predictions with shape (batch, pred_len, n_out_features).

    orig_dim : int
        Number of original observed series (OTS).

    ats_dim : int
        Number of auxiliary time series to generate.

    constructor : str
        One of 'linear', 'conv', 'attention'.

    input_key : Optional[str]
        If your training pipeline uses keyword args to pass the encoder input
        (e.g. x_enc=...), set input_key to that name. Otherwise the wrapper
        will assume the encoder input is the first positional argument.

    final_proj: bool
        If True, applies a learnable linear projection from full model output
        (n_out_features) -> orig_dim to obtain final predictions of shape
        (batch, pred_len, orig_dim). If the wrapped model already returns
        predictions with n_out_features == orig_dim, final_proj is a no-op.
    """

    def __init__(self, model: nn.Module, orig_dim: int, ats_dim: int = 4,
                 constructor: str = 'conv', input_key: Optional[str] = None,
                 final_proj: bool = True):
        super().__init__()
        self.model = model
        self.orig_dim = orig_dim
        self.ats_dim = ats_dim
        self.input_key = input_key
        self.constructor = constructor
        print(ATSConstructor)
        self.ats_constructor = ATSConstructor(orig_dim=orig_dim, ats_dim=ats_dim, constructor=constructor)
        self.final_proj_flag = final_proj

        # final projector from (orig_dim + ats_dim) -> orig_dim
        self.final_proj = nn.Linear(orig_dim + ats_dim, orig_dim) if final_proj else None

    def _inject_ats_into_args(self, args, kwargs):
        """
        Finds encoder input tensor from args/kwargs, appends ATS channels and
        returns new args/kwargs. Assumes encoder input has shape (b, t, orig_dim).
        """
        if self.input_key is not None and self.input_key in kwargs:
            x_enc = kwargs[self.input_key]
            ats = self.ats_constructor(x_enc)
            kwargs[self.input_key] = torch.cat([x_enc, ats], dim=-1)
            return args, kwargs

        # assume first positional arg is x_enc
        if len(args) >= 1:
            x_enc = args[0]
            ats = self.ats_constructor(x_enc)
            new_first = torch.cat([x_enc, ats], dim=-1)
            new_args = (new_first,) + args[1:]
            return new_args, kwargs

        raise RuntimeError("Could not find encoder input in args/kwargs to inject ATS."
                           " If your pipeline uses a different signature, set input_key.")

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, dict]:
        """
        Forwards inputs through ATS constructor + wrapped model, then projects
        output to original OTS dimensionality if needed.

        Returns
        -------
        pred_final : Tensor (batch, pred_len, orig_dim)
        regs : dict of regularization terms {'l1':..., 'diversity':..., 'continuity':...}
        """
        # 1) inject ATS into encoder input
        new_args, new_kwargs = self._inject_ats_into_args(args, kwargs)

        # 2) forward through the wrapped model
        model_out = self.model(*new_args, **new_kwargs)
        # allow wrapped model to return (pred, aux) or just pred
        if isinstance(model_out, tuple) or isinstance(model_out, list):
            pred_full = model_out[0]
            # keep extra outputs if any
        else:
            pred_full = model_out

        # pred_full expected shape: (b, pred_len, n_out)
        if pred_full.dim() != 3:
            raise ValueError(f"Wrapped model must return tensor shape (b,pred_len,n_out). Got {pred_full.shape}")

        b, pred_len, n_out = pred_full.shape

        # If model outputs exactly orig_dim, no projection is necessary
        if n_out == self.orig_dim:
            pred_final = pred_full
        elif n_out == self.orig_dim + self.ats_dim:
            # project last dimension back to orig_dim
            # apply same linear projection to each time step
            pred_final = self.final_proj(pred_full)
        else:
            # Sometimes the wrapped model might be configured differently
            # Fallback: if final_proj exists, try to map n_out -> orig_dim by a linear
            # applied to last-dimension. Create temporary linear if shapes mismatch
            if self.final_proj is not None and n_out != (self.orig_dim + self.ats_dim):
                # create a mapping layer on-the-fly (registered bufferless) - but better to warn
                raise ValueError(
                    f"Wrapped model output dim (n_out={n_out}) is incompatible with expected dims.\n"
                    f"Expected either orig_dim={self.orig_dim} or orig_dim+ats_dim={self.orig_dim + self.ats_dim}.\n"
                    "Please instantiate the wrapped model to output (orig_dim + ats_dim) features so CATS can project back,\n"
                    "or set final_proj=False if your model already returns orig_dim-sized output.")

        # 3) compute regularizers from current ATS batch (caller decides weights)
        regs = self._compute_regs()

        return pred_final, regs

    def _compute_regs(self) -> dict:
        """Compute L1 (sparsity), diversity (decorrelation), and continuity penalties.

        Note: these are computed from the constructor parameters and, for diversity,
        from a small synthetic pass using a dummy input if we cannot access the
        last generated ATS in this context. For efficiency and clarity we compute
        diversity from constructor weights as a proxy (authors use activations).
        If you prefer exact activation-based diversity, call ats_constructor on
        a representative batch and compute covariances externally in the training loop.
        """
        regs = {}
        # L1 on linear projection weights (sparsity)
        regs['l1'] = torch.norm(self.ats_constructor.proj.weight, p=1)

        # Diversity penalty: encourage off-diagonal smallness in proj weight correlations
        W = self.ats_constructor.proj.weight  # (ats_dim, orig_dim)
        # compute gram matrix (ats_dim x ats_dim)
        G = torch.matmul(W, W.t())  # (k,k)
        off_diag = G - torch.diag(torch.diag(G))
        regs['diversity'] = (off_diag ** 2).sum()

        # Continuity penalty: prefer smooth temporal kernels if conv constructor is used
        if hasattr(self.ats_constructor, 'temporal_conv'):
            # penalize high-frequency components of conv kernel by L2 on discrete derivative
            k = self.ats_constructor.temporal_conv.weight  # (out_ch, in_ch, kernel)
            # compute finite difference along kernel dim
            diff = k[..., 1:] - k[..., :-1]
            regs['continuity'] = (diff ** 2).sum()
        else:
            regs['continuity'] = torch.tensor(0., device=regs['l1'].device)

        return regs

# --------------------------------------------------
# Small helper: factory for easy instantiation
# --------------------------------------------------
def build_cats_wrapper(model: nn.Module, orig_dim: int, ats_dim: int = 4, constructor: str = 'conv',
                       input_key: Optional[str] = None, final_proj: bool = True) -> CATSWrapper:
    """Convenience factory.

    Example:
        wrapper = build_cats_wrapper(my_model, orig_dim=7, ats_dim=4, constructor='conv')
    """
    return CATSWrapper(model=model, orig_dim=orig_dim, ats_dim=ats_dim,
                       constructor=constructor, input_key=input_key, final_proj=final_proj)

# --------------------------------------------------
# Usage note (to include in file for developers):
# - Training loop should treat the wrapper as the model: it returns (pred, regs)
# - Compute main loss on pred and add weight * regs['l1'] etc.
# - Typical weights: l1_lambda 1e-4 .. 1e-2, diversity_lambda 1e-3 .. 1e-1,
#   continuity_lambda 1e-4 .. 1e-2. Tune per dataset.
# --------------------------------------------------