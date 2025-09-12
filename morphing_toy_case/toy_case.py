# toy_morphing_ridge_demo.py
# Morphing toy example and figures (PNG + PDF).
# Seeded & deterministic.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from pathlib import Path

# --------------------------
# 0) Config
# --------------------------
SEED = 7
T = 1000
LAG = 3                 # causal lag x -> y
LOOKBACK = 80           # rolling window for correlation (causal)
EMA_ALPHA = 0.20        # smoothing for correlation
ALPHA_M = 0.70          # morphing strength (approx +/- 70%)
MORPH_CLAMP = (0.30, 1.70)
RIDGE_ALPHA = 0.8
TRAIN_RATIO = 0.70
YLAGS, XLAGS, H = 5, 3, 1  # simple one-step forecaster
OUTDIR = Path("./morphing_toy_case")
OUTDIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

# --------------------------
# 1) Synthetic data with regimes
# --------------------------
t = np.arange(T)
# Regimes: (start, end, beta)
regimes = [
    (60, 150, -1.2),   # strong negative influence
    (180, 360, 1.6),   # strong positive influence
    (580, 820, 1.8),   # strong positive influence
]
beta = np.zeros(T)
for s, e, b in regimes:
    beta[s:e] = b

# Exogenous x: structured + noise
x = 0.8*np.sin(2*np.pi*t/50.0) + 0.25*np.sin(2*np.pi*t/17.0) + rng.normal(0, 0.5, T)

# Target y: AR(1) + seasonal + regime-dependent exogenous effect + noise
y = np.zeros(T)
phi = 0.55
season = 0.9*np.sin(2*np.pi*t/30.0)
for i in range(T):
    ar = phi*y[i-1] if i > 0 else 0.0
    exog = beta[i]*x[i-LAG] if i-LAG >= 0 else 0.0
    y[i] = 0.15 + ar + season[i] + exog + rng.normal(0, 0.45)

# --------------------------
# 2) Lag-aware causal rolling correlation -> morph ratio
# --------------------------
# Correlate y[t-L:t) with x[t-L-LAG:t-LAG) (causal alignment)
x_shifted = np.roll(x, LAG)
x_shifted[:LAG] = 0.0

corr = np.full(T, np.nan)
for i in range(T):
    start, end = i - LOOKBACK, i
    if start >= 0 and end > start:
        xs = x_shifted[start:end]
        ys = y[start:end]
        sx, sy = np.std(xs), np.std(ys)
        if sx > 1e-8 and sy > 1e-8:
            corr[i] = np.corrcoef(xs, ys)[0, 1]
        else:
            corr[i] = 0.0

# EMA smoothing for stability
corr_smooth = np.zeros_like(corr)
last = 0.0
for i, v in enumerate(np.where(np.isnan(corr), 0.0, corr)):
    last = EMA_ALPHA * v + (1 - EMA_ALPHA) * last
    corr_smooth[i] = last

# Map correlation -> morph ratio (sign-preserving power)
gamma = 0.8
corr_signed_mag = np.sign(corr_smooth) * (np.abs(corr_smooth) ** gamma)
morph_ratio = 1.0 + ALPHA_M * corr_signed_mag
morph_ratio = np.clip(morph_ratio, *MORPH_CLAMP)

# Apply morphing
x_morphed = morph_ratio * x

# --------------------------
# 3) Build lagged features & train Ridge forecasters
# --------------------------
def make_lagged(y, x, ylags=5, xlags=3, h=1):
    rows, targets = [], []
    start = max(ylags, xlags)
    for ti in range(start, len(y) - h):
        feats = [y[ti - l] for l in range(1, ylags + 1)]
        feats += [x[ti - l] for l in range(0, xlags)]  # include current x_t as l=0
        rows.append(feats)
        targets.append(y[ti + h])
    return np.asarray(rows), np.asarray(targets), start

X_base, Y, base_start = make_lagged(y, x, YLAGS, XLAGS, H)
X_morph, Y2, _ = make_lagged(y, x_morphed, YLAGS, XLAGS, H)
assert np.allclose(Y, Y2)

n = len(Y)
n_train = int(TRAIN_RATIO * n)
idx_train = np.arange(n_train)
idx_test = np.arange(n_train, n)

ridge_base = Ridge(alpha=RIDGE_ALPHA, random_state=0).fit(X_base[idx_train], Y[idx_train])
ridge_morph = Ridge(alpha=RIDGE_ALPHA, random_state=0).fit(X_morph[idx_train], Y[idx_train])

pred_base = ridge_base.predict(X_base[idx_test])
pred_morph = ridge_morph.predict(X_morph[idx_test])

mse_base = mean_squared_error(Y[idx_test], pred_base)
mse_morph = mean_squared_error(Y[idx_test], pred_morph)
improv = 100.0 * (mse_base - mse_morph) / mse_base

print(f"MSE (baseline): {mse_base:.4f}")
print(f"MSE (morphed) : {mse_morph:.4f}")
print(f"Relative improvement: {improv:.2f}%")

# Time indices aligned to target Y
time_for_targets = np.arange(base_start + H, base_start + H + n)
test_times = time_for_targets[idx_test]

# --------------------------
# 4) Plot helpers
# --------------------------
def shade_regimes(ax):
    for s, e, _ in regimes:
        ax.axvspan(s, e, alpha=0.15)  # default style, print-friendly

LW = 1.0  # thin strokes as requested

# Figure 1: y & x with regimes
fig1 = plt.figure(figsize=(10, 4))
ax = plt.gca()
ax.plot(y, label="Target $y_t$", linewidth=LW)
ax.plot(x, label="Exogenous $x_t$", linewidth=LW)
shade_regimes(ax)
ax.legend()
ax.set_title("Target and exogenous with relevant regimes shaded")
ax.set_xlabel("Time")
ax.set_ylabel("Value")
fig1_path = OUTDIR / "fig1_y_x_shaded.png"
plt.tight_layout()
plt.savefig(fig1_path, dpi=200)
plt.close(fig1)

# Figure 2: morph ratio
fig2 = plt.figure(figsize=(10, 3.5))
ax = plt.gca()
ax.plot(morph_ratio, label="Morph ratio $m(t)$", linewidth=LW)
shade_regimes(ax)
ax.legend()
ax.set_title(f"Morph ratio from lag-aware correlation (clamped [{MORPH_CLAMP[0]}, {MORPH_CLAMP[1]}])")
ax.set_xlabel("Time")
ax.set_ylabel("$m(t)$")
fig2_path = OUTDIR / "fig2_morph_ratio.png"
plt.tight_layout()
plt.savefig(fig2_path, dpi=200)
plt.close(fig2)

# Figure 3: x before/after morphing vs y
fig3 = plt.figure(figsize=(10, 4))
ax = plt.gca()
ax.plot(y, label="Target $y_t$", linewidth=LW)
ax.plot(x, label="Original $x_t$", linewidth=LW)
ax.plot(x_morphed, label="Morphed $\\tilde x_t$", linewidth=LW)
shade_regimes(ax)
ax.legend()
ax.set_title("Exogenous before/after morphing vs target")
ax.set_xlabel("Time")
ax.set_ylabel("Value")
fig3_path = OUTDIR / "fig3_x_before_after.png"
plt.tight_layout()
plt.savefig(fig3_path, dpi=200)
plt.close(fig3)

# Figure 4: test-only forecasts (no train shading)
fig4 = plt.figure(figsize=(10, 4))
ax = plt.gca()
ax.plot(test_times, Y[idx_test], label="True $y_t$ (test)", linewidth=LW)
ax.plot(test_times, pred_base, label="Pred (original exog)", linewidth=LW)
ax.plot(test_times, pred_morph, label="Pred (morphed exog)", linewidth=LW)
# Shade only regimes overlapping test
for s, e, _ in regimes:
    if e > test_times[0]:
        s_clamp = max(s, test_times[0])
        e_clamp = min(e, test_times[-1])
        if s_clamp < e_clamp:
            ax.axvspan(s_clamp, e_clamp, alpha=0.15)
ax.legend()
ax.set_title(f"Forecast comparison (test only) — MSE base={mse_base:.3f}, morphed={mse_morph:.3f} (Δ={improv:.1f}%)")
ax.set_xlabel("Time (test period)")
ax.set_ylabel("Value")
fig4_path = OUTDIR / "fig4_forecasts_test_only.png"
plt.tight_layout()
plt.savefig(fig4_path, dpi=200)
plt.close(fig4)

# Composite 4-panel (paper-ready): PNG + PDF
fig, axes = plt.subplots(4, 1, figsize=(10, 12))

# (a)
ax = axes[0]
ax.plot(y, label="Target $y_t$", linewidth=LW)
ax.plot(x, label="Exogenous $x_t$", linewidth=LW)
shade_regimes(ax)
ax.legend(loc="upper right")
ax.set_title("(a) Target and exogenous (relevant regimes shaded)")
ax.set_xlabel("Time"); ax.set_ylabel("Value")

# (b)
ax = axes[1]
ax.plot(morph_ratio, label="Morph ratio $m(t)$", linewidth=LW)
shade_regimes(ax)
ax.legend(loc="upper right")
ax.set_title("(b) Morph ratio from lag-aware rolling correlation")
ax.set_xlabel("Time"); ax.set_ylabel("$m(t)$")

# (c)
ax = axes[2]
ax.plot(y, label="Target $y_t$", linewidth=LW)
ax.plot(x, label="Original $x_t$", linewidth=LW)
ax.plot(x_morphed, label="Morphed $\\tilde x_t$", linewidth=LW)
shade_regimes(ax)
ax.legend(loc="upper right")
ax.set_title("(c) Exogenous before/after morphing vs target")
ax.set_xlabel("Time"); ax.set_ylabel("Value")

# (d) test-only
ax = axes[3]
ax.plot(test_times, Y[idx_test], label="True $y_t$ (test)", linewidth=LW)
ax.plot(test_times, pred_base, label="Pred (original exog)", linewidth=LW)
ax.plot(test_times, pred_morph, label="Pred (morphed exog)", linewidth=LW)
for s, e, _ in regimes:
    if e > test_times[0]:
        s_clamp = max(s, test_times[0])
        e_clamp = min(e, test_times[-1])
        if s_clamp < e_clamp:
            ax.axvspan(s_clamp, e_clamp, alpha=0.15)
ax.legend(loc="upper right")
ax.set_title(f"(d) Forecasts (test only) — MSE base={mse_base:.3f}, morphed={mse_morph:.3f} (Δ={improv:.1f}%)")
ax.set_xlabel("Time (test period)"); ax.set_ylabel("Value")

plt.tight_layout()
paper_png = OUTDIR / "paper_ready_composite.png"
paper_pdf = OUTDIR / "paper_ready_composite.pdf"
plt.savefig(paper_png, dpi=300)
plt.savefig(paper_pdf)
plt.close(fig)

print(f"\nSaved figures to: {OUTDIR.resolve()}")
print(f"Composite: {paper_png}, {paper_pdf}")
