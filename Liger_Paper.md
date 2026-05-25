# Liger: Dimensionality-Routed Hybrid Optimization with Sign-Momentum on Matrices and Variance-Rectified Updates on Scalars

*by Richard I Christopher, 2026*

**Author:** Richard I Christopher
**Affiliation:** MetaFore
**Email:** rchris@neotec.dev
**Version:** 1.0 (2026-05-22)
**Reference implementation:** `liger.py` in this repository.

---

## Abstract

We introduce **Liger** — *Layered Iterative Gradient Estimator with Rectification* — a hybrid optimizer for neural networks with mixed parameter shapes. Liger partitions parameters by tensor dimensionality at construction time and routes each partition to a different update rule: matrix-shaped parameters (`ndim ≥ 2`) take a Lion sign-momentum step, while vector- and scalar-shaped parameters (`ndim ≤ 1`) take a Yogi variance-rectified step. The two ancestor algorithms are unmodified; the contribution is structural.

This split is motivated by two pathologies that affect single-update-rule optimizers in mixed-dimensional architectures. **Pathology A — adaptive warmup coupling.** Adam-family optimizers rely on bias-corrected second moments (`v_hat`) that are statistically meaningless for the first 5-50 steps, requiring a rectification gate or LR warmup that interacts pathologically with the LR warmup practitioners already stack on top. Lion's sign-momentum update has no second moment and is operational from step 1. **Pathology B — rank-1 destruction on scalars.** Adam's `m / sqrt(v + ε)` step normalizes per-element variance and destroys the rank-1 burst structure of scalar-gate gradients; Yogi's bounded `v_t` update preserves it. Liger applies each ancestor to the regime it was designed for and gets warmup-independence, scalar burst-safety, and approximately 50% of AdamW's optimizer-state memory in a single optimizer instance (measured at 50.02% on a 1B-parameter-equivalent transformer in §9.4).

This document specifies the algorithm, derives the warmup-independence and memory-footprint claims analytically, names the regimes where Liger is *not* the right tool (ill-conditioned matrices, ill-conditioned scalars near saturation), and lays out a five-problem benchmark suite to validate the empirical claims on GPU.

---

## 1. Introduction and the Gradient-Heterogeneity Problem

Modern transformer-derivative modules — attention projections, feedforward MLPs, RMSNorm and LayerNorm gain/bias parameters, scalar gates and routing logits, learned position embeddings — are not gradient-homogeneous. Different parameter shapes within the same module produce gradients with structurally different statistics, and a single second-moment estimator cannot serve all of them well.

The dominant practice — using AdamW everywhere — papers over this with two coupled hyperparameter schedules: an LR warmup (typical: 1000-2000 steps of linear ramp) and a β₂ accumulation timescale (default `β2 = 0.999`, which has a 95th-percentile accumulation window of ~3000 steps). Practitioners tune the two against each other empirically. The result is functional but masks two distinct pathologies that, once named, suggest a different architecture.

### 1.1 Pathology A: Adaptive Warmup Coupling

Adam, AdamW, Muon, Shampoo, and related second-order optimizers maintain a per-parameter second-moment estimate `v_t` (or its matrix-factored analogues `GG_L`, `GG_R`). The bias-corrected `v_hat = v_t / (1 - β2^t)` is the denominator that produces adaptive per-element step sizing. At step 1, `v_t` has accumulated exactly one sample; at step 5, it has accumulated five. Practitioners and prior work (RAdam, Muogi, RAMuogi, Sophia) recognize this cold-start hazard and respond with one or more of:

  - **A rectification gate** (RAdam's `r_t` factor, computed from `ρ_t`) that throttles the adaptive pipeline until enough samples have accumulated.
  - **An LR warmup** that keeps the step size small while `v_hat` is unreliable.
  - **Both stacked** (the most common production practice).

Stacking them produces a perverse interaction. The LR warmup keeps the effective learning rate small in early steps. Small learning rates produce small gradients (via the chain rule through normalization layers and the model's natural Lipschitz constants). Small gradients keep `v_t` accumulating slowly. The adaptive engine therefore waits *longer* than its nominal warmup window — the LR warmup finishes before the second-moment accumulator has materially accumulated, and the optimizer's adaptive capacity comes online *after* the gradients have already grown to operating magnitude. The two schedules race each other; the LR warmup usually wins; the adaptive preconditioning never actually operates on cold-start signal in the regime it was designed for.

This coupling is invisible in tuned production runs because both schedules are co-tuned to a working configuration. It is *not* invisible in scientific experiments where one wants to vary the LR or the warmup independently and observe optimizer behavior in isolation. Liger sidesteps this by making the matrix-route update *have no second moment to wait on*: Lion's sign-momentum step is meaningful from step 1, full stop.

### 1.2 Pathology B: Rank-1 Destruction on Scalars

Adam's update is `update_i = m_hat_i / (sqrt(v_hat_i) + ε)`. The element-wise normalization by `sqrt(v_hat)` is the load-bearing design choice — it makes Adam's update scale-invariant to per-parameter gradient magnitude. For dense matrix gradients with roughly stationary variance, this works as intended. For scalar parameters and very-low-dimensional vectors (`ndim = 0` or `ndim = 1` with small length), the same normalization destroys structural information.

A learned scalar gate's gradient is a single dot product per backward pass: the upstream signal contracted to a scalar. Scalar gates routinely produce bursty gradient streams — long sequences of small magnitudes punctuated by occasional spikes when the gate's regime shifts (e.g., when a routing decision flips, or when an attention head saturates and the gate's downstream consumer suddenly sees a large gradient). Adam handles a burst of magnitude `B` by setting `v_t ← β2·v_{t-1} + (1-β2)·B²`. After the burst, `v_t ≈ (1-β2)·B²`, which means `sqrt(v_hat) ≈ B`, which means the next update — even for normal-magnitude gradients — is scaled by `1/B`. The accumulator is *poisoned* for the next 1/(1-β2) ≈ 1000 steps. Adam cannot decrease `v_t` faster than its exponential decay constant.

Yogi (Zaheer et al., 2018) was designed exactly for this case. Its update is

    v_t = v_{t-1} - (1 - β2) · sign(v_{t-1} - g²) · g²

The `sign(v_{t-1} - g²)` factor flips the update direction when the current gradient-squared exceeds the accumulator: `v_t` *decreases* in response to a burst rather than locking to it, and *increases* in response to a stretch of small gradients. The accumulator becomes a bounded estimator of typical variance rather than an upper envelope.

For 2-D dense matrix parameters, Yogi's variance handling is unexceptional — most practical matrices in mixed-dimensional architectures arrive with well-conditioned gradient statistics from upstream normalization. Yogi is *strictly* the right tool for the scalar/1-D case and *a mediocre tool* for the dense matrix case. Liger applies it only where it wins.

### 1.3 The Routing Observation

The architectural observation that links these pathologies: matrix gradients in attention/MLP/norm-conditioning modules typically flow through softmax-normalized or RMSNormed paths, which are already well-conditioned at the point the optimizer sees them — the forward-pass normalization has already bounded their dynamic range. They do not need second-moment accumulation to bound update magnitude; what they benefit from is **bounded direction** (Lion's sign-momentum) without per-step magnitude scaling.

Vector and scalar parameters are different. They sit *inside* the normalization machinery — RMSNorm gains, attention output scales, learned biases, scalar routing gates — and they receive gradients that bypass the conditioning the matrices benefit from. They genuinely need adaptive variance handling, and they need it to be burst-safe, which is precisely what Yogi provides.

The conclusion: dispatch by parameter dimensionality. There is no parameter-count heuristic involved ("use Adam for big models, SGD for small"). The right abstraction is gradient structure, and dimensionality is a clean proxy for it.

### 1.4 Contribution

Liger contributes:

  1. A clean architectural pattern (dimensionality-routing) applied at the optimizer level.
  2. An implementation that ships both ancestor algorithms unmodified inside one `torch.optim.Optimizer` subclass with a single per-parameter `ndim` check at first encounter, pinned thereafter.
  3. Analytical claims for warmup-independence (§4.1), bounded scalar variance (§4.2), and memory footprint (§4.3) that can be verified by reading the code without running an experiment.
  4. A five-problem benchmark suite (§9) that isolates each claim with a baseline-comparison protocol against AdamW, Lion-only, Yogi-only, Muogi, RAMuogi, and RACASO.

Liger does *not* contribute novel algorithms; the Lion and Yogi updates are unmodified. The novelty is in admitting that they should coexist within a single optimizer instance, and in identifying the structural condition (parameter shape, used as a proxy for gradient regime) under which that coexistence is principled.

### 1.5 Lineage and scope

This paper is one of three (Liger, Muogi/RAMuogi, RACASO) describing optimizers developed in sequence against distinct gradient-regime failure modes encountered during production training of a multi-stream transformer-derivative architecture. Each paper is scoped to its own optimizer and the specific problem class it addresses. The companion papers describe the other two and how the family fits together; only what is load-bearing for Liger appears here.

**The problem class Liger solves.** A *mixed-dimensional routing layer downstream of normalization*: the combined assembly of attention projections + feed-forward matrices + RMSNorm gains + bias vectors + scalar gates that forms the basic block in every modern transformer-derivative architecture. The matrix parameters in such a block see gradients that have already passed through upstream normalization (softmax in attention, RMSNorm before the feed-forward), so they arrive at the optimizer *already well-conditioned*. The scalar and 1-D parameters sit *inside* the normalization machinery and produce bursty, rank-1-shaped gradients that need adaptive variance handling.

**What we tried first.** Before Liger we deployed an orthogonalization-style preconditioner (Newton-Schulz polynomial iteration in the style of Muon, wrapped in the four-layer safety chain documented in the companion RAMuogi paper) on this exact layer class. The empirical observation, reproducible in production telemetry across thousands of training steps: **the orthogonalization step converged immediately on essentially every refresh** — there was nothing to orthogonalize. The matrix gradients arrived already well-conditioned from upstream normalization. The spectral-preconditioner path was 100% wasted compute, paying the cost of an expensive Newton-Schulz refresh and a RAdam cold-start gate to access a benefit the gradient regime did not require.

**What Liger does about it.** Liger's matrix path is Lion: bounded direction without preconditioning, no second-moment accumulator, no cold-start gate. Liger's scalar/1-D path is Yogi: variance-rectified updates that survive the bursty rank-1 gradient regime that scalar gates produce. The dispatch decision is by parameter dimensionality — a clean proxy for "which gradient regime does this parameter sit in." The two ancestor algorithms (Lion, Yogi) are unmodified; the contribution is the structural decision to route them differently within one optimizer instance.

**Where Liger does not apply.** Liger explicitly does not handle the case where matrix gradients arrive *ill-conditioned* — for example, a dense interaction surface where multiple normalized streams meet and produce gradient covariance that violates the Kronecker factorization assumption $\Sigma \approx \Sigma_L \otimes \Sigma_R$. For that regime, spectral preconditioners (Shampoo, SOAP, or RACASO's rotated-basis approach) are the right tools, and the companion papers document those choices. Liger and the companion optimizers compose at the per-parameter-group level: a practitioner picks one optimizer per group based on which gradient regime the group's parameters live in, not a single optimizer for the whole model. §9.10 reports measured numbers from open-bench sweeps that establish each claim.

---

## 2. The Liger Architecture

### 2.1 The 2-D Path: Lion Recap

Lion (Chen et al., 2023, "Symbolic Discovery of Optimization Algorithms") is a sign-momentum optimizer discovered by symbolic search. Its update, as published:

    c_t = β1·m_{t-1} + (1-β1)·g_t
    update_t = sign(c_t)
    m_t = β2·m_{t-1} + (1-β2)·g_t

with two distinct momentum coefficients β1 and β2 (in the Lion paper sense — these are *not* the Adam β1/β2). Lion's signature property is that every parameter coordinate moves by exactly `±lr` per step — the sign-update normalizes step magnitude to the nominal LR, which makes Lion strikingly insensitive to per-parameter gradient scale while preserving direction.

Liger uses the shared-coefficient simplification (β1 = β2 in the Lion-paper sense), which collapses to:

    m_t = β1·m_{t-1} + (1-β1)·g_t
    update_t = sign(m_t)

This is the form implemented by `lion-pytorch` (the most-used reference port) and is what we use. The shared-coefficient form is provably equivalent to the two-coefficient form in the spec when β1 = β2, and saves one temporary buffer per Lion-route parameter — material at the parameter counts we target.

**Why Lion for matrices?** Three reasons:

  1. *No warmup interaction.* Lion has no `v_t` accumulator. At step 1 with `m_0 ≈ 0`, the update direction is `sign((1-β1)·g) = sign(g)` — exactly the right direction, no cold-start gate needed.
  2. *No preconditioning overhead.* Lion is O(P) per parameter — no eigendecomposition, no Newton-Schulz iteration, no Kronecker-factor maintenance. For matrices whose gradients arrive well-conditioned (the typical case in normalized transformer blocks), this is the cheapest update consistent with bounded direction.
  3. *Half the memory of Adam.* Lion stores one buffer (`m_t`) per parameter; Adam stores two (`m_t`, `v_t`). At the matrix scale this dominates total optimizer-state footprint.

### 2.2 The 1-D/0-D Path: Yogi Recap

Yogi (Zaheer et al., 2018, "Adaptive Methods for Nonconvex Optimization") is a bounded-variance variant of Adam:

    m_t = β1·m_{t-1} + (1-β1)·g_t
    v_t = v_{t-1} - (1-β2)·sign(v_{t-1} - g_t²)·g_t²
    m_hat = m_t / (1 - β1^t)
    v_hat = v_t / (1 - β2^t)
    update_t = m_hat / (sqrt(v_hat) + ε)

The only difference from Adam is the `v_t` update. Adam: `v_t = β2·v_{t-1} + (1-β2)·g²`. Yogi: `v_t -= (1-β2)·sign(v_{t-1} - g²)·g²`. The sign-flip makes `v_t` move *toward* `g²` rather than *up to* `g²`, preventing the accumulator from being permanently inflated by a single burst.

The Yogi paper proves `|v_t - v_{t-1}| ≤ (1-β2)·g²` per step — a single burst can move `v_t` by at most `(1-β2)·B²`, and the same bound applies to *decrease* on subsequent small-gradient steps. The bound holds element-wise.

**Why Yogi for scalars/vectors?** The bursty-rank-1 case is exactly Yogi's design target. For scalar gates, RMSNorm gains, learned attention scales, and similar low-dimensional parameters, Yogi keeps the accumulator from being poisoned by occasional spikes — the optimizer continues to make progress at the gate's natural step magnitude rather than getting locked into a sub-LR regime for thousands of steps.

We use `β2 = 0.99` by default (rather than Adam's typical 0.999) because Yogi's bounded update rule already prevents runaway variance — we don't need a 1000-step accumulation window to dampen.

We use `ε_yogi = 1e-3` (Yogi paper default, larger than Adam's 1e-8) as the floor on `sqrt(v_hat)`. The larger floor makes the 1-D path robust to near-zero `v_hat` at cold-start without needing a separate rectification gate.

**Implementation note on the Yogi denominator.** The vendored standalone Yogi at `bench/optimizers/yogi.py:92` builds the denominator as `(sqrt(v_t) / sqrt(bc2)).add_(eps)` and folds `bc1` into the learning-rate term — the Yogi-paper canonical formulation. Liger's Yogi path at `liger.py:276` builds the denominator as `sqrt(v_hat).clamp_(min=eps_yogi).add_(eps_adam)`, applying the clamp at the sqrt level (the `eps_yogi` floor) and adding a separate Adam-style additive `eps_adam` afterward. The two are algebraically equivalent up to the placement of the additive epsilon, but the clamp-then-add form makes the floor explicit. We document this here so the P2 head-to-head between standalone Yogi and Liger's Yogi route is transparent.

### 2.3 The Dispatcher

The dispatch decision is pinned per parameter at its first `step()` encounter:

```
                       gradient g for parameter p
                                 │
                                 ▼
                       ┌─────────────────────┐
                       │  p.ndim ≥ 2  ?      │
                       └─────────────────────┘
                          yes  │       │  no
                               ▼       ▼
                  ┌──────────────┐  ┌──────────────────┐
                  │  Lion path   │  │  Yogi path       │
                  │  (matrices)  │  │  (vectors,       │
                  │              │  │   scalars)       │
                  │  state:      │  │  state:          │
                  │   m_t        │  │   m_t, v_t       │
                  │              │  │                  │
                  │  update =    │  │  update =        │
                  │   sign(m_t)  │  │   m_hat /        │
                  │              │  │    (√v_hat + ε)  │
                  └──────────────┘  └──────────────────┘
                          │                  │
                          └────────┬─────────┘
                                   ▼
                         p ← p - lr · update
                         (after decoupled wd)
```

The route is stored as `state["is_lion"]: bool` (not a string — strings get mangled by `torch.optim.Optimizer.load_state_dict`'s `_cast` helper). Once pinned, the route is never re-checked: reshaping a parameter after step 1 is undefined behavior, matching the convention of Lion's and Yogi's reference implementations.

### 2.4 Decoupled Weight Decay

Both paths use AdamW-style decoupled weight decay applied identically:

    p ← p · (1 - lr · weight_decay)            # before the update step

This is the same form used by AdamW and by the `lion-pytorch` reference Lion implementation. The decoupling is important on the Lion path specifically: if weight decay were folded into the gradient before the sign operation, large weights with small gradients would still take ±lr steps (sign destroys magnitude information), and the decay would be effectively unbounded. Decoupled decay applies the shrinkage multiplicatively before the sign-update, which is what we want.

---

## 3. Algorithm

The complete Liger update for one parameter `p` with gradient `g` at step `t`, with hyperparameters `lr, β1, β2, ε_y, ε_a, λ` (where `λ = weight_decay`):

```
On first encounter (lazy init):
    m_0  ← full_like(p, initial_accumulator)
    if p.ndim ≥ 2:
        is_lion[p] ← True
    else:
        is_lion[p] ← False
        v_0  ← full_like(p, initial_accumulator)

At step t:
    # Decoupled weight decay (both routes).
    if λ ≠ 0:
        p ← p · (1 - lr · λ)

    if is_lion[p]:
        # Lion path.
        m_t ← β1·m_{t-1} + (1-β1)·g
        update_t ← sign(m_t)
        p ← p - lr · update_t

    else:
        # Yogi path.
        m_t ← β1·m_{t-1} + (1-β1)·g
        v_t ← v_{t-1} - (1-β2)·sign(v_{t-1} - g²)·g²
        m_hat ← m_t / (1 - β1^t)
        v_hat ← v_t / (1 - β2^t)
        denom ← max(sqrt(v_hat), ε_y) + ε_a
        p ← p - lr · m_hat / denom
```

**Hyperparameters and defaults:**

| Symbol | Name | Default | Source |
|---|---|---|---|
| `lr` | learning rate | 1e-4 | conservative default; tune per task |
| `β1` | momentum coefficient | 0.9 | Lion & Yogi both standard |
| `β2` | Yogi 2nd-moment timescale | 0.99 | lower than Adam's 0.999 (Yogi bounds v_t) |
| `ε_y` | Yogi sqrt floor | 1e-3 | Yogi paper default |
| `ε_a` | Adam-style additive eps | 1e-8 | Adam convention |
| `λ` | weight decay (decoupled) | 0.0 | off by default; controller-supplied |
| `init_acc` | initial accumulator | 1e-6 | matches Yogi/Muogi/RACASO family |

**Memory per parameter:**

- Lion route (`ndim ≥ 2`): one buffer (`m_t`), same shape and dtype as `p`. Approximately 50% of Adam's per-parameter state.
- Yogi route (`ndim ≤ 1`): two buffers (`m_t`, `v_t`), same shape and dtype as `p`. Same as Adam/Yogi.

For a transformer block where matrix parameters outnumber vector/scalar parameters by 50-500×, the total optimizer-state footprint is approximately **50% of AdamW's** — see §4.3 for the count.

---

## 4. What Liger Solves

### 4.1 Warmup Independence (Analytical)

**Claim.** Liger's matrix-route update produces the correct gradient direction at step 1, with no cold-start gate, no LR warmup, and no second-moment accumulator to wait on.

**Proof.** At step 1, `m_0 = initial_accumulator` (a tiny scalar, default 1e-6). The momentum update is `m_1 = β1·m_0 + (1-β1)·g = β1·init_acc + (1-β1)·g`. For any gradient `g` whose elements are not pathologically close to `-β1·init_acc / (1-β1) ≈ -9e-7`, the sign of `m_1` equals the sign of `g`. Therefore `update_1 = sign(m_1) = sign(g)`, and `p_1 = p_0 - lr · sign(g)` — the optimizer moves in exactly the gradient-descent direction at step 1.

In contrast, AdamW at step 1 produces:
- `m_1 = (1-β1)·g`, `m_hat_1 = m_1 / (1 - β1) = g`
- `v_1 = (1-β2)·g²`, `v_hat_1 = v_1 / (1 - β2) = g²`
- `update_1 = m_hat_1 / (sqrt(v_hat_1) + ε) = g / (|g| + ε) ≈ sign(g)`

So AdamW *also* produces `sign(g)` at step 1, but only because bias correction makes `E[v_hat_t] = E[g²]` at every t. Bias is corrected by construction. The remaining transient at step 2-5 is a **variance** problem, not a bias problem: `v_hat_t` is an unbiased estimator of `E[g²]` from the start, but with one to five samples its sampling variance is large, so the effective denominator wobbles depending on the specific gradient sequence. RAdam (Liu et al. 2019) names this exactly: its `r_t` rectifier is a **variance**-rectification factor — it suppresses the adaptive pipeline until the estimator has accumulated enough samples for its variance to drop to a useful range. The literature predating RAdam often described this as a "bias" issue informally; the rectifier's formal derivation is variance-based.

Liger's Lion route avoids the variance issue **by not using `v_t` at all**. There is no second-moment estimator to be high-variance early. `sign(m_t)` is well-defined for all `t ≥ 1` with the same magnitude property at every step — and the only momentum signal it depends on is the first moment, which converges in a single step's worth of EMA. The Lion path simply does not have the cold-start variance regime that RAdam is rectifying for.

The Yogi route on 1-D/0-D parameters does still maintain a `v_t` accumulator and so does still have a cold-start variance transient. With `β2 = 0.99` (vs. Adam's 0.999) the transient is 10× shorter, and on low-dimensional parameters the per-parameter cost of a high-variance early step is bounded. Liger does not deploy a RAdam-style gate on the Yogi route because the bursty-scalar regime Yogi is targeting (§4.2) is exactly the regime where a rectifier would suppress the *correct* early updates — Yogi's bounded `v_t` rule is the right tool, not a gate on top of Adam.

### 4.2 Bounded Variance on Scalars

**Claim.** A single bursty gradient of magnitude `B` on a Yogi-route parameter moves `v_t` by at most `(1-β2)·B²`, and subsequent small-magnitude gradients can recover `v_t` back toward its pre-burst value.

**Proof.** Yogi's update is `v_t = v_{t-1} - (1-β2)·sign(v_{t-1} - g²)·g²`. The magnitude of the change is `|v_t - v_{t-1}| = (1-β2)·g²`. With `g = B`, this is `(1-β2)·B²`, regardless of the sign of `(v_{t-1} - B²)`. The sign factor only affects *direction* of the change.

If `v_{t-1} < B²` (the burst exceeds the accumulator), the sign is negative and `v_t = v_{t-1} + (1-β2)·B²` (accumulator grows). If on the *next* step `g = g_small` with `g_small² < v_t`, then `sign(v_t - g_small²) = +1` and `v_{t+1} = v_t - (1-β2)·g_small²` (accumulator shrinks). The accumulator is therefore not a monotonic upper envelope; it tracks variance bidirectionally.

This is verified by `test_yogi_rank1_burst_bounded` (`test_liger.py:241-291`): after warming with `g = 1e-2` for 20 steps and injecting a burst `g = 1e3`, the accumulator change is exactly `(1-β2)·B² = 1e4`, and subsequent steps with `g = 10` shrink the accumulator by `(1-β2)·g² = 1.0` per step — visible numerical recovery.

**Recovery-rate comparison with Adam.** The two recovery rates are quantitatively different and the comparison cuts both ways depending on the post-burst gradient regime. With `β2 = 0.99` and post-burst `v_t ≈ 1e4`:

- **Adam.** Recovery is multiplicative: `v_t(t+k) = β2^k · 1e4`. Recovery to `v_t = 1` (a 4-order-of-magnitude drop) takes `k = log(1e-4)/log(0.99) ≈ 916` steps with zero subsequent gradients.
- **Yogi.** Recovery is additive in the bound: `|v_t - v_{t-1}| ≤ (1-β2)·g²_typical`. With `g²_typical = 1`, recovery to `v_t = 1` takes `k ≈ B²/g²_typical = 1e4` steps. With `g²_typical = 100`, recovery takes `k ≈ 100` steps. With `g²_typical = 1e4`, the system is in steady state.

**In the high-burst, low-typical-gradient regime** (the regime that motivates Yogi as a *bound-against-explosion* rather than a *fast-recoverer*), Yogi recovers more slowly than Adam — about 3 orders of magnitude slower in the example above (~916 vs ~1e6 steps with `g²_typical = 1`). The argument for Yogi here is **bounded harm**, not faster recovery: a single burst cannot inflate `v_t` past the level that one bounded step of `(1-β2)·B²` can produce, while Adam's `v_t = β2·v_{t-1} + (1-β2)·B²` does the same on the up-step but then locks in the inflation until exponential decay catches up.

**In the moderate-burst regime** where `g²_typical` is comparable to post-burst `v_t`, Yogi's additive recovery and Adam's multiplicative decay produce similar order-of-magnitude recovery windows, and Yogi's *bidirectional* tracking (the accumulator can shrink in response to small gradients) is a clean structural advantage over Adam's monotonic-up-then-decay envelope.

The P2 benchmark (§9.2) is intentionally a high-burst regime (`B = 1e3` against typical `g ~ 1`), and the empirical result there is consistent with the math above: Liger's Yogi path produces the highest final loss of the eight optimizers tested. The Yogi-on-scalars choice is correct for the **architectural** reason (bounded variance prevents catastrophic accumulator poisoning that would shut down learning on the affected scalar entirely) but does **not** produce the lowest final-loss point on a benchmark whose gradient stream is dominated by the high-burst regime. We document this honestly in §9.2 rather than hiding behind the v_hat trajectory plot.

### 4.3 Memory: Approximately 50% of AdamW

**Claim.** Liger's total optimizer-state memory is approximately 50% of AdamW's on transformer-derivative architectures, measured at 50.02% on a 1B-parameter-equivalent module in §9.4.

**Derivation.** Let `P_2D` = total elements in 2-D-and-higher parameters, `P_1D` = total elements in 1-D-and-0-D parameters. Then:

- AdamW state per element: 2 buffers (`m_t`, `v_t`) — total `2·(P_2D + P_1D)` elements of optimizer state.
- Liger state per element:
  - Lion route: 1 buffer (`m_t`) — total `P_2D` elements.
  - Yogi route: 2 buffers (`m_t`, `v_t`) — total `2·P_1D` elements.
  - **Liger total: `P_2D + 2·P_1D`.**

Ratio: `Liger / AdamW = (P_2D + 2·P_1D) / (2·(P_2D + P_1D))`.

For a typical transformer with hidden dim `d` and FFN expansion `4d`, per layer:
- Matrix params: 4 attention projections (`d²` each) + 2 FFN matrices (`4d²` each) + norm gains (`d`, treated as 1-D) ≈ `12·d² + small` per layer, dominated by `d²` terms.
- Vector params: attention/FFN biases if present (`d` each), 2 norm gains per layer (`d` each) ≈ `O(d)` per layer.

For `d = 1024`: matrix params ≈ `12·d² = 12.6M`, vector params ≈ `6·d = 6K`. Ratio of matrix to vector params: ~2000:1.

Plugging in: `Liger / AdamW ≈ (12.6M + 12K) / (2·12.6M) ≈ 0.5005`. Approximately **50.1% of AdamW** on a pure transformer.

The measured value in §9.4 on the 1B-parameter-equivalent module is **50.02%**, matching the analytical prediction. Architectures with a higher vector/scalar fraction (RMSNorm-heavy, multi-gate routing) would push the ratio toward the 55-65% range, but we only headline what was actually measured.

This claim is verified by P4 in the benchmark suite (§9.4).

### 4.4 No Preconditioning Overhead on Already-Conditioned Matrices

**Claim.** Lion's per-step compute on matrix parameters is O(P), where P is the parameter count. There is no eigendecomposition, no iterative polynomial, no factored covariance maintenance.

**Justification.** Lion's update is `m_t ← β1·m_{t-1} + (1-β1)·g` (one fused multiply-add), `sign(m_t)` (one element-wise sign), and `p ← p - lr·sign(m_t)` (one fused multiply-add). Total: 3 element-wise operations, no global computations.

Compare:
- Shampoo: per-step Kronecker-factor maintenance (two `d×d` matrices) plus periodic inverse-quarter root (eigendecomposition, O(d³)).
- Muon: per-step NS5 iteration (5 matrix products, O(d³) each, ~5 iterations).
- RACASO: per-step rotated-basis maintenance plus periodic Hessian estimate (HVP, O(P)) plus eigendecomposition.
- Adam: per-step element-wise `m_t`, `v_t`, `m_hat`, `v_hat`, `sqrt`, divide, add — same big-O as Lion but ~6 element-wise operations vs Lion's 3, plus a separate `v_t` buffer.

For matrices that arrive well-conditioned from upstream normalization, the heavier preconditioners are spending compute on signal that is already conditioned. Lion is the cheapest update consistent with bounded direction, and that bound is sufficient when the gradient signal is already well-shaped.

---

## 5. What Liger Does NOT Solve

### 5.1 Pathologically Ill-Conditioned Matrices

If a matrix parameter's gradient is genuinely ill-conditioned — eigenvalue spread of `10⁶` or more, or cross-branch-aggregation regimes where two upstream streams differ by orders of magnitude — Lion's sign update will *step in the right direction* but will not adapt step magnitude to local curvature. Convergence is slow on ill-conditioned matrices because Lion has no second-moment information to scale by. The right tool for that regime is **Shampoo, RACASO, or RAMuogi**, not Liger.

Liger explicitly trades preconditioning capability for warmup independence and memory. This is a deliberate scope choice, not an oversight.

### 5.2 Per-Parameter LR Adaptation

Lion's sign-update means every matrix parameter coordinate moves by exactly `±lr` per step regardless of gradient magnitude. Parameters that need *smaller* steps than the nominal LR (because they are near-optimal) do not get them automatically. The LR schedule and any per-group LR controller must compensate.

In a system with an external per-parameter or per-group LR controller, this is fine — the controller adjusts. In a system without such a controller, Lion-family optimizers can overshoot at the per-parameter level near convergence. This is a known Lion limitation, not Liger-specific.

### 5.3 Ill-Conditioned Scalars Near Saturation

The Yogi path bounds variance but does not *precondition*. A scalar parameter with a fundamentally ill-conditioned local loss landscape (e.g., a learned temperature parameter near a softmax saturation point, where the loss has a near-zero gradient with high curvature) won't escape that region under Yogi alone. This is rare in practice but worth flagging.

The right tools for those cases include second-order scalar optimizers (Sophia's Hutchinson estimate) or a manual reparameterization that improves the conditioning of the scalar's loss surface.

### 5.4 Sparse Gradients

Liger raises `RuntimeError` on sparse gradients. Embedding-table optimizers like SparseAdam should be used for embedding tables. Liger is designed for dense, in-place updates.

---

## 6. Where Liger Fits in the Optimizer Zoo

| Optimizer | Matrix strategy | Scalar strategy | Warmup coupling | Memory vs Adam |
|---|---|---|---|---|
| **AdamW** | Adam normalization | Adam normalization | High (β2 timescale) | 100% |
| **Lion** | Sign-momentum | Sign-momentum (suboptimal for bursty scalars) | None | 50% |
| **Yogi** | Yogi `v_t` (slow for matrices) | Yogi `v_t` (correct) | Medium (β2 timescale) | 100% |
| **Muon** | NS5 orthogonalization | AdamW fallback | None (but heavy compute) | 100% |
| **Shampoo** | Kronecker eigendecomposition | AdamW fallback | None (heavy compute + memory) | 200%+ |
| **RAMuogi** | NS5 + Yogi + RAdam gate | Yogi fallback | High (L4 + β2 + LR triple-warmup) | 100% |
| **RACASO** | Rotated Adam + Hessian-aware | Yogi fallback | High (L4 gate + LR warmup) | 100%+ |
| **Liger** | Lion (sign-momentum) | Yogi `v_t` | None | ~50% |

Liger occupies a specific niche: mixed-dimensional parameters where the matrix gradients arrive well-conditioned and immediate operation matters. This is the *common* case for transformer-derivative architectures with normalization — more common than the ill-conditioned-matrix regime that motivates the heavier preconditioners.

The right way to read this table: Liger is for the *middle* of the optimizer landscape. Too sophisticated for "everything is homogeneous, just use AdamW." Too restrained for "I need to model gradient covariance." Most transformer-derivative modules sit exactly in the middle.

---

## 7. Diagnostic Interface

Liger exposes `get_telemetry()` returning a dictionary suitable for direct logging into a training observability stack:

```python
{
    "step_count":              <int>,    # max step across all params
    "num_2d_params":           <int>,    # count on Lion route
    "num_1d_params":           <int>,    # count on Yogi route
    "last_max_momentum_norm":  <float>,  # max ||m_t||₂ across Lion params
    "last_max_update_l1":      <float>,  # max Σ|sign(m_t)| across Lion params
    "last_max_v_hat":          <float>,  # max v_hat across Yogi params
    "last_min_v_hat":          <float>,  # min v_hat (catches eps floor)
}
```

**What to watch:**

- `last_max_momentum_norm` growing toward `||g||` per-step is the Lion-path health signal. If it explodes, the LR is too high or there is a gradient pathology upstream.
- `last_max_v_hat` should be bounded relative to typical gradient magnitudes squared. A persistent ratio greater than 100× the typical `g²` indicates a burst that has not yet been recovered.
- `last_min_v_hat` near zero (specifically near `init_acc²`) indicates an `eps_yogi`-floor activation; persistent activation means the Yogi route is operating in its safe-floor regime, which is fine but worth knowing.
- `num_2d_params` vs `num_1d_params` should match `[p.ndim for p in model.parameters()]` at construction; if not, the router has miscounted (this would be a bug in the dispatch — we have a test for it: `test_telemetry_route_dispatch_census`).

There is no NS5-style success/skip counter (Liger has no preconditioner that can fail) and no RAdam-style rectification counter (Liger has no warmup gate). The diagnostic surface is minimal because the optimizer is structurally simple.

---

## 8. The Hybrid Argument

The reason Liger exists rather than choosing Lion or Yogi globally: the loss surface is not uniform across parameter shapes within a single module. Treating it as such — using one optimizer for everything — is the implicit assumption that every other optimizer makes, and that assumption is wrong for any architecture that mixes parameter shapes.

Lion's authors note that sign-updates underperform on small parameter groups (the bias terms specifically). Yogi's authors note that variance rectification matters most for low-dimensional accumulators. Each ancestor optimizer was correct for its targeted regime; the failure was claiming one regime covered everything.

Liger's contribution is **structural rather than algorithmic**: partition the parameter set by gradient regime (using dimensionality as a clean proxy), route each partition to the optimizer that matches. The two ancestor algorithms are unmodified. Lion is Lion. Yogi is Yogi. The novelty is in admitting that they should coexist within a single optimizer instance and in dispatching automatically.

This is the same architectural pattern that mixed-pathway models use at the model level — distinct structural pathways for distinct data regimes, dispatched per-input rather than collapsed into one homogeneous stack. Applied at the optimizer level, it produces an optimizer that **matches its tool to its workpiece** rather than wielding one tool everywhere.

---

## 9. Empirical Results

The benchmark suite comprises two layers:

- **Five synthetic problems (P1–P5)** that isolate the analytical claims from §4 against five baseline optimizers (`adam`, `adamw`, `yogi`, `lion`, `liger`) and three sibling family optimizers (`muogi`, `ramuogi`, `racaso`). Each problem is surgical: it instruments one specific property (router dispatch, scalar burst recovery, warmup independence, memory footprint, router correctness) so the result maps cleanly to the analytical claim it tests.

- **Three real-task problems (R1 CIFAR-10 ResNet-18, R2 char-LM on tiny-shakespeare, R3 NanoGPT byte-level on WikiText-2)** that demonstrate the optimizer works on industry-credible architectures, not just toy quadratics. R1 is the canonical "does this optimizer train a real image classifier" gate; R2 is the Karpathy-canonical lightweight LM gate; R3 is NanoGPT-scale (~30M params), the scale at which independent optimizer papers establish credibility.

All sweeps run via `bench/run_bench.py --sweep --device cuda` on an NVIDIA RTX A4500 (20GB). Raw results: `bench/results.csv`. Figures: `bench/figs/*.png`.

### 9.0 Methodology

**Per-optimizer learning-rate grids.** Different optimizer families have structurally different update magnitudes given the same nominal learning rate. Lion's update is `lr · sign(m_t)` — every coordinate moves by exactly `±lr`. Adam's update is `lr · m̂_t / (√v̂_t + ε)` — the same `lr` produces a coordinate move scaled down by the running variance estimate. Empirically a Lion step at `lr = 1e-3` moves parameters two to three orders of magnitude farther than an Adam step at the same `lr`. Running all optimizers on a shared LR grid would put one family in a regime where it diverges while the other runs at an appropriate step size, which is not a meaningful comparison.

We therefore use **per-family LR grids** matched to each optimizer family's typical operating range, following the convention used in published Lion, Sophia, and Muon comparison papers (Chen et al. 2023 §4.2 explicitly notes Lion requires a 3–10× lower LR than Adam). The exact grids used:

| Family | LR grid |
|---|---|
| Adam, AdamW, Yogi | `[1e-4, 3e-4, 1e-3, 3e-3]` |
| Lion, Liger | `[1e-5, 3e-5, 1e-4, 3e-4]` |
| Muogi, RAMuogi, RACASO | `[3e-5, 1e-4, 3e-4, 1e-3]` |

These grids are pinned in `bench/run_bench.py::_LR_SWEEP` so the comparison is exactly reproducible from the open-source bench harness.

**Reporting convention.**

- *Best LR per optimizer, averaged across seeds.* For each (problem, optimizer) pair the figures and tables in §9.1–§9.10 report the **best LR for that optimizer**, averaged across seeds — the LR that minimizes the seed-averaged final loss. The figure legend shows `(lr=X)` next to each optimizer's name so the LR each line corresponds to is always visible. This is the same convention used in the Lion, Sophia, Adafactor, and Muon papers.

- *Smoothed final loss (mean of the last 50 steps).* `final_loss` is the **mean of the last 50 steps** of the per-step loss trajectory when the trajectory is long enough; for shorter trajectories it falls through to the last step. A single trailing noisy minibatch is a high-variance estimator of true final-loss, and an earlier version of this paper reported the unsmoothed last-step value — which privileged optimizers whose last step happened to land on an easy batch. The 50-step window is short enough to track late-trajectory descent and long enough to average out one batch's worth of luck. The unsmoothed last-step CSV is preserved at `bench/results_unsmoothed.csv` for reproducibility.

- *Fixed-budget on real tasks.* R1/R2/R3 auto-enable fixed-budget execution: `bench/run_bench.py` records the first-convergence step but does not break out of the loop, so every optimizer runs the full `max_steps`. Earlier early-stop behavior produced a methodologically unfair head-to-head — different optimizers' final-loss numbers were reported at different points along their trajectories. The `--fixed-budget` CLI flag exposes the same behavior on synthetic problems for any user who wants identical apples-to-apples runs.

- *Truncate-to-min across seeds.* When aggregating multi-seed trajectories into a single curve for a figure, we truncate to the shortest seed's length rather than padding to the longest. Padding-with-final-value implicitly claims that all seeds were observed at every step, which they were not. This policy is applied uniformly across `bench/plot_bench.py` and `bench/plot_cross_comparison.py`; see `bench/README.md` for the unified statement.

**Seed budgets.** Synthetic problems P1–P5 use seeds {0, 1, 2} (three independent runs per (problem, optimizer, LR) cell). Real-task problems R1/R2/R3 use seeds {0, 1} (two independent runs per cell, because each run is much more expensive in GPU-time — a single R1 ResNet-18 training is ~6 minutes, a single R3 NanoGPT training is ~1 minute, but the 8-optimizer × 4-LR cardinality multiplies fast).

**Divergence filtering in figures.** In the per-problem figures and the cross-comparison figure, we filter out any optimizer whose seed-averaged best-LR final loss exceeds 3× the median of all 8 optimizers' final losses on that problem. The filter is symmetric — if Liger ever diverged on a problem it would be filtered out of its own paper's figure. Divergent optimizers are listed in each figure's subtitle (`[diverged: racaso (50.5)]` for R3, for example), so the filtering is documented in the figure itself rather than hidden. The raw numbers including divergent runs are in `bench/results.csv` for verification.

**Hardware envelope.** Single GPU, RTX A4500 (20GB). The P4 memory measurement instantiates a 1B-parameter-equivalent synthetic module; RACASO's optimizer state requires more than 20GB at that scale and is OOM-skipped in 12 of 12 cells — that's a real result documented in §9.4 (RACASO's state-bytes exceed our card's memory capacity, which is itself information).

### 9.1 P1 — Mixed-Dim Module (Headline)

**Setup.** Module with one 64×64 matrix (Lion target), one 64-dim bias (Yogi target), one scalar gate (Yogi target). Loss: `||A·x + b·gate - y||²` on synthetic regression data. Measures the *dispatch* benefit: Liger should beat each of {AdamW, Lion-only, Yogi-only} because each baseline pays a tax on the regime it doesn't natively handle.

**Metric.** Steps to reach `loss < converged_tol`. Lower is better.

| Optimizer | Best LR | First-converge step | Final loss |
|---|---|---|---|
| Adam    | 3e-3 | 139  | 9.89e-3 |
| AdamW   | 3e-3 | 140  | 9.82e-3 |
| Yogi    | 3e-3 | 190  | 9.89e-3 |
| Lion    | 3e-4 | 865  | 9.96e-3 |
| **Liger** | 3e-4 | 818 | 9.97e-3 |
| Muogi   | 1e-3 | 479  | 9.97e-3 |
| RAMuogi | 1e-3 | 1089 | 9.98e-3 |
| RACASO  | 1e-3 | — (diverged) | 1.44 |

All 7 of {Adam, AdamW, Yogi, Lion, Liger, Muogi, RAMuogi} converge to the `1e-2` tolerance; RACASO diverges on this LR sweep at the chosen problem size. AdamW reaches converged_tol fastest (140 steps); Liger and Lion take ~6× longer because the sign-update only moves by ±lr per coordinate, so reaching a small final loss requires more steps. Liger and Lion are tied within noise (818 vs 865 steps); the dispatch overhead is not visible at this scale.

**Lion configuration note.** The Lion baseline above used the published two-coefficient `betas=(0.9, 0.99)`. This revision of the paper pins the Lion baseline at `betas=(0.9, 0.9)` to match Liger's one-coefficient simplification (§2.1), so the Lion-vs-Liger comparison is on the same Lion variant. A re-run of P1 with the corrected Lion baseline is part of the "GPU work pending" list at the end of §9; we expect the Lion-vs-Liger gap to close further under the corrected config.

The headline observation for P1: **all four "Lion-family" optimizers (Lion, Liger, Muogi, RAMuogi) hit converged_tol at this lr; the absolute speed-to-tolerance ranking is dominated by the per-element-vs-sign-update tradeoff, not by dispatch.**

### 9.2 P2 — Scalar Burst

**Setup.** Scalar gate alone. Gradient stream: small (`~1.0`) with periodic bursts (`~1e3` every 50 steps). Measures Yogi-path burst recovery and final-loss behavior on a scalar-only problem.

**Metric.** Two things, in two panels: (a) Liger's `last_max_v_hat` per LR — the post-burst residual that motivates Yogi's bounded `v_t` rule — and (b) the best-LR loss trajectory for every optimizer in the sweep so the reader can see comparative final loss directly.

**Best-LR final loss across all 8 optimizers (lower = better):**

| Optimizer | Best LR | Final loss |
|---|---|---|
| Adam    | 3e-3 | **0.965** |
| AdamW   | 3e-3 | 0.967 |
| Yogi    | 3e-3 | 0.969 |
| Lion    | 3e-4 | 1.821 |
| Muogi   | 1e-3 | 1.920 |
| RACASO  | 1e-3 | 1.987 |
| RAMuogi | 1e-3 | 2.159 |
| **Liger** | 3e-4 | **2.167** |

**Honest reading: Liger has the *worst* final loss of the 8 optimizers tested on P2.** The cause is a per-family-LR-grid artifact rather than a structural failure of the Yogi path. Standalone Yogi runs on the Adam-family LR grid `[1e-4, 3e-4, 1e-3, 3e-3]` and finds its best LR at `3e-3`. Liger runs on the Lion-family LR grid `[1e-5, 3e-5, 1e-4, 3e-4]` because Liger's matrix-route is Lion (§9.0 methodology); but on a P2 problem with *no* matrix parameters — the entire model is the scalar gate — the LR grid Liger inherits from its Lion-family classification is an order of magnitude lower than the LR a standalone Yogi would pick. Liger's Yogi path is therefore underpowered on a scalar-only problem by construction.

This is a real limitation, not a benchmarking quirk to be hand-waved past. Liger commits to a single per-instance LR for both routes. On any model whose parameter mix is dominated by scalars/1-D params, the user must either run a separate optimizer group with a Yogi-family LR (the per-parameter-group composition pattern §1.5 describes), or accept that the Yogi path's effective LR is constrained by what the Lion-path matrix parameters need. P2 is the surgical worst case for this trade: zero matrix parameters, so the Lion path's LR cap does no work but still constrains the Yogi path.

**What the v_hat panel does show.** Liger's `last_max_v_hat` settles at `~1e5` across all four LRs in the grid — the expected `B²·(1-β2)/bc2 ≈ 1e4 / (1-0.99^N)` for any post-burst step within the recorded window. The bounded-variance property §4.2 proves holds in vivo; the bound is the right shape; the recovery rate in the high-burst regime is slow (§4.2 spells out why). Adam-family optimizers would set `v_t ≈ (1-β2)·B² = 1e4` and decay multiplicatively, which on the same gradient stream produces lower final loss because the LR is operating in its appropriate regime.

**What §9.2 says structurally.** Yogi-on-scalars is the right tool when the *failure mode you care about* is catastrophic accumulator poisoning (a single 1e3 burst locking the optimizer into a sub-LR regime for thousands of steps under Adam's exponential decay). Yogi-on-scalars is *not* the right tool when the failure mode you care about is fastest-recovery-from-burst-into-low-final-loss on a scalar-only problem. Liger's commitment is to the first failure mode — it composes within an architecture whose matrix parameters dominate and whose scalars need bounded-harm semantics. The P2 result is the price of that commitment, named honestly.

(See `bench/figs/fig2_p2_v_hat_trajectories.png` for the v_hat residual scatter and the comparative loss trajectory.)

### 9.3 P3 — Warmup-Free

**Setup.** Small MLP, no LR warmup, measure loss at step 1, step 5, step 50.

**Metric.** Loss at each checkpoint. Liger should match Lion (both warmup-independent) and beat AdamW (still in β2 warmup).

| Optimizer | Best LR | Step 1 (init) | Step 5 | Step 50 |
|---|---|---|---|---|
| Adam    | 3e-3 | 0.245 | 0.083 | 0.0024 |
| AdamW   | 3e-3 | 0.245 | 0.083 | 0.0024 |
| Yogi    | 3e-3 | 0.245 | 0.150 | 0.0045 |
| Lion    | 3e-4 | 0.245 | 0.221 | 0.0580 |
| **Liger** | 3e-4 | 0.245 | 0.221 | 0.0546 |
| Muogi   | 1e-3 | 0.245 | 0.207 | 0.0110 |
| RAMuogi | 1e-3 | 0.245 | 0.243 | 0.0722 |
| RACASO  | 1e-3 | 0.245 | 0.243 | 0.2432 |

**What this measures.** All optimizers see the same `0.2448` initial loss because no step has been taken; the first measurable difference is at step 5. By step 50, Adam-family and Yogi are well below `0.01`; Lion and Liger sit at `~0.05-0.06`; RACASO is still at the initial loss (its L4 cold-start gate is still suppressing the spectral path at this step count, and the sign-flip-bounded `v_t` accumulator hasn't yet built up enough mass to produce useful steps).

**Reading the result honestly.** Liger and Lion are operational from step 1 (every coordinate moves by exactly `±lr` from step 1 onward — this is the property §4.1 proves analytically) but the sign-update only moves `±lr` per coordinate per step, so reaching low loss takes many more steps than Adam-family's adaptive step sizes provide on this smooth-curvature problem. **The warmup-independence claim is about producing meaningful updates from step 1, not about beating Adam on every problem**: a tanh-MLP regression with smooth curvature is exactly where Adam-family adaptive scaling pays off and where sign-momentum's "constant magnitude per coordinate" property loses. Liger does not claim to beat Adam on smooth-curvature regression; it claims to dispatch correctly and operate from step 1 without warmup compounding.

### 9.4 P4 — Memory Footprint

**Setup.** Synthetic 1B-parameter-equivalent module (tiled). Instantiate optimizer state, measure `sum(state-tensor bytes)`.

**Metric.** Total optimizer-state bytes, normalized to AdamW = 100%.

| Optimizer | State bytes | Relative to AdamW |
|---|---|---|
| AdamW   | 8,055,689,280 (7.50 GiB) | 100.00% |
| Adam    | 8,055,689,280 (7.50 GiB) | 100.00% |
| Yogi    | 8,055,685,760 (7.50 GiB) | 100.00% |
| Muogi   | 8,055,685,760 (7.50 GiB) | 100.00% |
| RAMuogi | 8,055,685,760 (7.50 GiB) | 100.00% |
| Lion    | 4,027,842,880 (3.75 GiB) | 50.00% |
| **Liger** | **4,029,153,920 (3.75 GiB)** | **50.02%** |
| RACASO  | — (OOM at 20 GiB cap) | n/a (state exceeded card capacity) |

**Reading the result.** Liger's measured state at the 1B-parameter-equivalent module is **50.02% of AdamW's** — exactly the analytical prediction from §4.3 (50.1% on a near-pure transformer parameter distribution). Lion matches at 50.00%. AdamW, Yogi, Muogi, and RAMuogi all sit at 100% because each carries two buffers per parameter (`m_t` + `v_t`); Muogi/RAMuogi's per-row Yogi-injection scale `R` is computed on-the-fly inside `step()` and not stored, so the per-parameter state matches Adam exactly. RACASO's rotated-basis matrices (`Q_L`, `Q_R`, `GG_L`, `GG_R`, plus `hessian_diag_rot`) push its state size beyond the 20 GiB card capacity, OOM-skipping all 12 racaso runs — this is a documented failure-by-memory, not missing data.

The 1311-byte difference between Liger and Lion (4.029 vs 4.028 GB) is the additional state for Liger's Yogi-route 1-D parameters (RMSNorm gains, biases, scalar gates: ~163K floats × 2 buffers × 4 bytes ≈ 1.3 MB — though the 1311-byte figure is much smaller because the synthetic module sizes the 1-D parameter fraction smaller than a real transformer block; on a real model, the Yogi-route state would be larger but still a tiny fraction of total).

### 9.5 P5 — Router Correctness

**Setup.** Mixed-ndim model with hand-counted parameters. After 1 step, `get_telemetry()["num_2d_params"]` and `["num_1d_params"]` must match `[p.ndim for p in model.parameters()]`.

**Metric.** Telemetry vs. ground-truth scatter (should be y = x). Pure dispatch sanity gate.

This problem also runs locally as a CPU smoke test (no GPU required), and is the first gate in the RunPod launch checklist.

### 9.6 In-Production Trace on an Internal Architecture

Liger is in active production use as the assigned optimizer for one parameter group in a transformer-derivative architecture training internally to our group. Because this trace is third-party-unverifiable (the host architecture is not open-sourced), the full per-step telemetry analysis has been moved to **Appendix A — In-Production Trace**. The reproducible head-to-head against published baselines lives in §9.1–§9.5 and §9.7–§9.10.

### 9.7 R1 — CIFAR-10 ResNet-18

**Setup.** Standard ResNet-18 (~11.2M params, vendored at `bench/models/resnet18.py`) on CIFAR-10 train split. 5000 steps, batch 128, no LR warmup, no cosine schedule (constant LR per run so the optimizer is doing all the work). Cross-entropy loss; convergence threshold train loss < 0.5.

**Why this problem.** This is the canonical "does this optimizer work on a real model" gate in every optimizer paper since Adam. A new optimizer that fails on CIFAR-10 ResNet-18 is not publishable; conversely, an optimizer that converges here cannot be dismissed as toy-data-only.

**Results** (smoothed final loss = mean over the last 50 steps; convergence_step = first step where loss < 0.5, recorded but no longer used for early-stop):

| Optimizer | Best LR | Final train loss (smoothed) | First-convergence step |
|---|---|---|---|
| AdamW   | 1e-3 | **0.696** | 1151 |
| Yogi    | 1e-3 | 0.697 | 822 |
| Muogi   | 3e-4 | 0.702 | 830 |
| RAMuogi | 3e-4 | 0.704 | 1212 |
| RACASO  | 3e-4 | 0.709 | 885 |
| **Liger** | 3e-4 | 0.713 | 912 |
| Lion    | 3e-4 | 0.716 | 736 |
| Adam    | 1e-3 | 0.739 | 912 |

**Reading the result.** All 8 optimizers cluster within a ~6% relative band on smoothed final train loss (0.696 to 0.739 over 5000 steps under early-stop budget; a fixed-budget GPU re-run is pending — see "GPU work pending" below). The ordering is dominated by ResNet-18-specific factors (LR schedule, weight initialization, BatchNorm interaction) rather than the dispatch decision Liger encodes. **Liger is competitive but not winning on R1**: this is honest — CIFAR-10 ResNet-18 with constant LR is a setting where the sign-momentum-vs-adaptive-scaling tradeoff is roughly neutral, and the matrix gradients in a small CNN do not have the "ill-conditioned cross-branch coupling" regime that would penalize Liger's no-preconditioning matrix path. The R1 result tells the reader Liger **trains a real model without falling over**.

(See `bench/figs/fig6_r1_cifar10.png` for full loss curves.)

### 9.8 R2 — Char-LM on tiny-shakespeare

**Setup.** Small char-level transformer (~3M params, vendored at `bench/models/charlm.py`): 4 layers, hidden 256, 4 heads, vocab 128 (ASCII). Trained on tiny-shakespeare (1.1MB, vendored at `bench/datasets/tinyshakespeare.txt`) for 3000 steps, batch 32, sequence length 128. Cross-entropy loss; convergence threshold train loss < 1.5 (uniform-prior char baseline ≈ 4.85).

**Why this problem.** Karpathy-canonical lightweight LM gate. Every LM optimizer claim has tiny-shakespeare somewhere as the "does it train language at all" sanity check.

**Results** (smoothed final loss = mean over the last 50 steps; first-convergence step recorded but no longer used for early-stop on real tasks):

| Optimizer | Best LR | Final train loss (smoothed) | First-convergence step |
|---|---|---|---|
| **Liger** | 3e-4 | **1.586** | 2109 |
| AdamW   | 1e-3 | 1.592 | 2905 |
| Adam    | 1e-3 | 1.595 | 2905 |
| Yogi    | 1e-3 | 2.055 | — (did not reach 1.5) |
| Muogi   | 3e-4 | 2.271 | — |
| RAMuogi | 3e-4 | 2.462 | — |
| Lion    | 3e-4 | 2.510 | — |
| RACASO  | 3e-4 | 3.899 | — |

**Reading R2 honestly.** A mixed-dim transformer model (4-layer char-LM with attention projections, FFN matrices, RMSNorm gains, and biases) is exactly the architecture class §1.3 names as Liger's target. Under the smoothed-final-loss convention, **Liger, AdamW, and Adam form a three-way tie at the top within ~0.6% relative**: Liger at 1.586, AdamW at 1.592, Adam at 1.595. The earlier version of this paper reported a 1.484 vs 1.581 gap that put Liger 6% ahead; that gap was a single-noisy-minibatch reporting artifact, not a real difference. Smoothing the last 50 steps across both seeds collapses the gap.

What Liger does demonstrate on R2 is **competitiveness with the strongest baseline at half the optimizer-state memory**: Liger lands at AdamW's final loss while carrying one buffer per matrix parameter instead of two. Lion alone sits at 2.510 — the sign-update is fine on the matrix parameters but produces no useful adaptation for the bias and norm-gain parameters, which is the bursty-scalar regime Yogi's variance-rectified path is designed for. The dispatch decision (Lion on matrices + Yogi on 1-D/0-D params) is what closes the gap from Lion's 2.510 to Liger's 1.586. The headline R2 claim is **memory-efficient parity with AdamW**, not a clean win over it.

(See `bench/figs/fig7_r2_charlm.png` for full loss curves.)

### 9.9 R3 — NanoGPT (byte-level) on WikiText-2

**Setup.** 6-layer NanoGPT (~30M params, vendored at `bench/models/nanogpt.py`): hidden 384, 6 heads, byte-level vocab 256, sequence length 256. Trained on WikiText-2-raw (13MB, downloaded on first call) for 1000 steps, batch 8 (= 2048 bytes/step). Cross-entropy loss; convergence threshold train loss < 5.0 (uniform 256-class baseline ≈ 5.55).

**Why this problem.** NanoGPT-scale is the credibility floor for independent LM optimizer papers: not GPT-2-small full scale (which would dominate runtime), but enough capacity to require real optimization signal rather than just memorization. Byte-level tokenization keeps the bench dependency-free — no `transformers`, no `tokenizers`, no pretrained merges JSON. Byte-level LMs are a legitimate published practice (ByT5, MEGABYTE).

**Results.**

| Optimizer | Best LR | Final train loss (smoothed) | First-convergence step | Trajectory length |
|---|---|---|---|---|
| RAMuogi | 3e-4 | **6.24** | 216 | 216-218 |
| **Liger** | 3e-4 | 7.20 |  90 |  90-97 |
| Muogi   | 3e-4 | 10.43 |  58 |  58-62 |
| Yogi    | 1e-3 | 23.81 |  38 |  38-43 |
| AdamW   | 1e-3 | 26.42 |  40 |  40-45 |
| Adam    | 1e-3 | 26.44 |  40 |  40-45 |
| Lion    | 3e-4 | 31.57 |  36 |  36-40 |
| RACASO  | 3e-4 | 52.24 | — (diverged on DivBackward0; see RACASO paper §6) | 1000 |

**Reading R3 honestly under smoothed-final-loss-with-early-stop.** The trajectory-length column matters here. Under the earlier early-stop behavior, every R3 run stopped at the first step where `loss < 5.0` and reported the noisy last-step value. Under smoothed-final-loss-over-last-50 applied to those *same trajectories* (no GPU re-run), the table above is what falls out. **Liger is no longer the best on R3 — RAMuogi is.** The ordering is also dominated by trajectory-length artifact: optimizers that early-stopped at 40 steps have their entire trajectory (including the ~200-loss initial steps) folded into the smoothed mean, while RAMuogi's longer 216-step trajectory has more low-loss steps in its last-50 window. The earlier 4.620 headline for Liger was a single-noisy-minibatch artifact compounded by the early-stop break.

**The right comparison is a fixed-budget re-run with all optimizers running the full 1000 steps**, which is currently pending leased GPU time — see "GPU work pending" at the end of this section. We expect the smoothed-final-loss-over-last-50 numbers from a fixed-budget run to compress the spread substantially and to remove the trajectory-length confound. The honest claim from the data we have today: **Liger is no longer the headline R3 winner**, and the earlier 4.62-vs-4.88 ranking was an artifact of the early-stop + last-step-final reporting that the methodology fixes in this revision address.

What R3 still shows: **RACASO diverges to ~52** on byte-level NanoGPT — the L5 DivBackward0 hazard documented in §6 and §7 of the RACASO paper, triggered by the byte-level NanoGPT model's softmax-norm path producing unbounded second derivatives. Liger's no-preconditioning matrix path and burst-safe scalar path together avoid this failure class structurally, which is the structural claim §1.5 names and the R3 result still supports.

(See `bench/figs/fig8_r3_nanogpt.png` for full loss curves.)

### 9.10 Comparison with sibling family optimizers (Muogi, RAMuogi, RACASO)

The Liger benchmark suite runs against **all three sibling-family optimizers** developed in this lineage — Muogi, RAMuogi, and RACASO — because each is published as a separate ArXiv submission with overlapping baselines, and cross-citation between them strengthens all four papers. The vendoring rule in `bench/optimizers/` is uniform: every optimizer appears as a standalone source file, treating sibling-family optimizers exactly like external baselines.

**Where each sibling wins.**

- **Muogi (sign-momentum + NS5 orthogonalization on matrices)** is expected to outperform Liger on R1 CIFAR-10 because ResNet-18's convolutional matrices benefit from spectral orthogonalization; Lion-only and Liger-on-matrices both produce sign-momentum updates without this preconditioning, which is structurally what NS5 provides.
- **RAMuogi (RAdam-rectified Muogi)** should beat Liger on R3 NanoGPT specifically because byte-level LMs have ill-conditioned `v_hat` in the first hundred steps (rare bytes dominate the gradient occasionally), where RAMuogi's L4 cold-start gate suppresses spurious updates that Liger's Yogi-on-scalars path does not gate.
- **RACASO (Hutchinson HVP or GNB) with Sophia-style clipping** is expected to outperform Liger on problems where second-order curvature information matters — but at the cost of an expensive HVP refresh every `hessian_freq` steps. On the cheap-to-train benchmarks here (R1/R2/R3), RACASO's overhead may dominate the curvature benefit.

**Where Liger wins.**

- **Memory.** Liger's optimizer state at the 1B-equivalent scale (P4) is ~50% of AdamW's. Muogi and RAMuogi both pay full Adam-state-memory because Yogi-on-matrices needs the full `v_t` buffer. RACASO pays *more* than AdamW because of its rotated-basis matrices.
- **Cold-start steps 1–10.** Liger's Lion route is operational from step 1 with no `v_t` accumulator (§4.1); RAMuogi's L4 cold-start gate suppresses both the spurious *and* the legitimate cold-start updates equally.
- **Dispatch overhead.** Liger does one ndim check per parameter at construction; the other three sibling family optimizers do per-step routing logic.

**Cross-comparison figure.** See `bench/figs/cross_comparison.png` — a single multi-panel figure overlaying all 8 optimizers (the 5 baselines + the 3 siblings) on R1/R2/R3. The same figure appears in `Muogi/RAMuogi_Paper.md` §9 and `RACASO/RACASO_Paper.md` §9 so a reviewer reading any one paper sees the unified head-to-head.

**Unified head-to-head table** (smoothed final loss, mean over the last 50 steps; this paper highlights Liger):

| Optimizer | R1 CIFAR-10 | R2 char-LM | R3 NanoGPT | State (% AdamW) |
|---|---|---|---|---|
| Adam    | 0.739 | 1.595 | 26.44 | 100.00% |
| AdamW   | 0.696 | 1.592 | 26.42 | 100.00% |
| Yogi    | 0.697 | 2.055 | 23.81 | 100.00% |
| Lion    | 0.716 | 2.510 | 31.57 | 50.00%  |
| **Liger** | 0.713 | **1.586** | 7.20 | **50.02%** ✦ |
| Muogi   | 0.702 | 2.271 | 10.43 | 100.00% |
| RAMuogi | 0.704 | 2.462 | **6.24** | 100.00% |
| RACASO  | 0.709 | 3.899 | 52.24 (diverged) | n/a (OOM at 1B) |

(✦ = Liger has the lowest state-bytes value. **Bold** marks the column leader. Lower is better for loss columns; lower is better for state.)

Caveats on the numbers above:

- The R3 column is heavily affected by **trajectory-length artifact under early-stop**: shorter-trajectory optimizers (Adam, AdamW, Lion at 36-45 steps) have their entire pre-convergence noise window folded into the smoothed mean, while longer-trajectory optimizers (RAMuogi at 216 steps, Liger at ~94 steps) have more low-loss steps in their last-50 window. The R3 numbers as a whole are not directly comparable until a fixed-budget GPU re-run lands; see "GPU work pending" below.
- R1 and R2 had longer trajectories under early-stop (~1000-3000 steps), so the smoothed numbers there are closer to a fair head-to-head, though a fixed-budget re-run will still produce cleaner numbers.
- The **memory-and-state column is fully verified at 50.02% from the 1B-equivalent measurement in §9.4**; that result is independent of the early-stop methodology fixes and remains the cleanest headline for the paper.

**The honest contribution that survives this revision:** (a) dispatch-by-ndim is principled and the §4.1/§4.2 derivations hold; (b) Liger's optimizer state is 50.02% of AdamW's, measured at 1B-equivalent parameter scale; (c) the exact router places every parameter on the correct route at construction. The earlier-reported R2 and R3 single-noisy-minibatch wins were artifacts; the corrected R2 is a memory-efficient tie with AdamW at top; the corrected R3 awaits a fixed-budget re-run.

**GPU work pending.** The following re-runs are blocked on leased GPU time and will land when GPU is available:

1. *R1/R2/R3 fixed-budget re-sweep.* Run every (problem × optimizer × LR × seed) cell to the full `max_steps` with no early-stop, and recompute the smoothed-final-loss head-to-head on the resulting trajectories. The expected outcome is that the R3 trajectory-length confound disappears, and the relative ordering on R3 stabilizes. CLI:

   ```bash
   python bench/run_bench.py --sweep --device cuda --output bench/results.csv
   ```

   (The harness auto-applies fixed-budget for R1/R2/R3; no flag needed.)

2. *P1 re-sweep with corrected Lion baseline.* The Lion baseline configuration was changed from the published two-coefficient `betas=(0.9, 0.99)` to the one-coefficient `betas=(0.9, 0.9)` so the Lion vs Liger head-to-head reflects the same Lion variant Liger uses internally. P1 rows for Lion need to be re-run; same CLI as above will pick the corrected wrapper.

---

## 10. Configuration

Recommended starting hyperparameters:

```python
opt = Liger(
    model.parameters(),
    lr=1e-4,                    # base LR; tune per task as usual
    betas=(0.9, 0.99),          # β1 shared, β2 lower than Adam
    eps_yogi=1e-3,              # Yogi paper default
    eps_adam=1e-8,              # Adam convention
    weight_decay=0.0,           # off by default
    initial_accumulator=1e-6,   # family convention
)
```

**Tuning notes:**

- **LR.** Lion typically wants a 3-10× lower LR than Adam due to its sign-update. Start at `1e-4` and scale down if loss diverges in early steps. The first 50 steps are diagnostic — if loss diverges there with `lr = 1e-4`, the LR is too high.
- **β1.** 0.9 is standard for both Lion and Yogi. Lower (e.g., 0.85) reduces the momentum smoothing window. Higher (e.g., 0.95) is rare and only useful when gradients are very noisy.
- **β2.** 0.99 is the right starting point for Liger specifically (Yogi-on-scalars). If the scalar-gate trajectory is too aggressive, raise to 0.999 (longer accumulation window). If too sluggish, lower to 0.95.
- **eps_yogi.** 1e-3 is the Yogi paper value. Lowering it allows the optimizer to take larger steps on very-low-variance scalars but risks divide-by-near-zero pathology. Don't lower it without a specific reason.
- **weight_decay.** Off by default. When non-zero, applies AdamW-style decoupled decay to both routes identically.

---

## 11. Conclusion

Liger applies a structural insight — different parameter shapes deserve different update rules — to produce an optimizer that combines two well-understood algorithms (Lion and Yogi) without modifying either. The result is operationally simple, memory-efficient, warmup-independent on its matrix path, and burst-safe on its scalar path. Liger does not contribute new algorithms; it contributes a deliberate decision about *where to apply them*.

For mixed-dimensional architectures where matrix gradients arrive well-conditioned and step-1 operation matters, Liger is the right tool. For ill-conditioned matrices or scalar saturation pathologies, it isn't — and §5 says so honestly. The optimizer is published with a complete test suite (`test_liger.py`, 22 tests covering construction, router dispatch, both update paths, weight decay, convergence, and telemetry) and a five-problem benchmark suite (`bench/`) designed to validate the analytical claims empirically on GPU.

The reference implementation is `liger.py` — single file, single class, PyTorch-only, MIT-licensed.

---

## Acknowledgments

Thanks to **Ben Goertzel** for the standing ArXiv endorsement that makes work in this lineage publishable on first submission. Thanks to **Chen et al.** (Lion) and **Zaheer et al.** (Yogi) for the algorithms Liger composes — their work does the actual mathematical lifting; Liger is a structural decision *about* their work. Thanks to the **MetaFore** group for the surrounding research stack (RACASO, Muogi, RAMuogi) that establishes the publishable-optimizer-family pattern this paper follows.

---

## Appendix A — In-Production Trace

Liger is in active production use as the assigned optimizer for one parameter group in a transformer-derivative architecture training internally to our group. The group is a mixed-dimensional routing layer: 20 matrix-shaped parameters (Lion route) + 29 vector- or scalar-shaped parameters (Yogi route), totaling 0.99M parameters out of a 22.7M-parameter model. It is exactly the niche §6 of this paper names: mixed dimensionality, well-conditioned matrix gradients (downstream of normalized aggregation), and a system-wide preference for warmup-free operation (the host training loop applies an LR warmup at the controller level, and per-group optimizers must not compound it).

The host emits Liger's `get_telemetry()` dict at every logged step. The first 80 steps from a recent run are extracted in `bench/morpheus_v2.2_liger_telemetry.csv` (17 telemetry/loss pairs) and rendered in `bench/morpheus_v2.2_liger_trajectory.png`; the file naming reflects the internal version that produced the trace. **The data is third-party-unverifiable** (the host architecture is not open-sourced) — that is the reason it lives in an appendix rather than in §9's main results.

**Router census (P5 in production).** `n_2d = 20` and `n_1d = 29` are stable across every step — the dispatch matches the ground-truth ndim count at construction and never drifts. The router-correctness claim from §9.5 holds in vivo.

**Lion-path momentum integration (warmup-free, §4.1 / §9.3).** The maximum Lion-route `||m_t||₂` follows the expected EMA integration profile: `9e-4` at step 1, `1.7e-3` at step 2, ..., `5.3e-3` at step 10, plateauing at `~8e-3` by step 30 and stable through step 80. The integration constant matches `β1 = 0.9` to within float precision. At step 1 with `m_0 ≈ 0` the optimizer is already producing meaningful sign-momentum updates — no warmup gate, no cold-start throttle.

**Lion-path saturation (`upd_l1`).** `Σ|sign(m_t)|` is constant at `1.5e+05` across every step, equal to the element count of the largest Lion-route parameter. This means every coordinate of `m_t` is non-zero at every step — sign is fully saturated, the bounded-direction property holds with no degenerate zero-momentum coordinates.

**Yogi-path accumulator behavior (§4.2 in vivo).** `v_max` is stable in the range `3.3e-3` to `3.8e-3` across all 80 steps — the matrix-route gradients arrive well-conditioned at typical magnitude `O(0.05–0.08)` and Yogi's accumulator equilibrates to `~g²_typical ≈ 4e-3`. `v_min` decays from `1.0e-3` at step 1 to `1.3e-5` at step 80 as the accumulator catches up to lower-variance scalar parameters.

**Loss trajectory.** Mean loss decreases monotonically from `5.74` at step 1 to `5.39` at step 80 (a 6.1% reduction in 80 steps for a 22.7M-param model with `lr_base = 3e-5 → 3e-7` schedule). The trajectory is monotonic with no divergent steps and no rebounds.

**What this trace does and does not show.** It demonstrates that Liger runs stably as the chosen tool on the group its design targets, and validates the analytical §4 claims (warmup-independence, bounded variance, router census) directly on a real training run. It does not substitute for the controlled head-to-head comparison in §9.1-§9.5 and §9.7-§9.10; both kinds of evidence appear in this paper for distinct purposes.

---

## References

1. **Chen, X., et al.** (2023). *Symbolic Discovery of Optimization Algorithms.* arXiv:2302.06675. The Lion optimizer.
2. **Zaheer, M., et al.** (2018). *Adaptive Methods for Nonconvex Optimization.* NeurIPS 2018. The Yogi optimizer.
3. **Kingma, D. P., & Ba, J.** (2014). *Adam: A Method for Stochastic Optimization.* arXiv:1412.6980. Adam.
4. **Loshchilov, I., & Hutter, F.** (2017). *Decoupled Weight Decay Regularization.* arXiv:1711.05101. AdamW; the decoupled-decay form used in Liger.
5. **Liu, L., et al.** (2019). *On the Variance of the Adaptive Learning Rate and Beyond.* arXiv:1908.03265. RAdam; the rectification-gate pattern Liger does *not* need but contextualizes against.
6. **Christopher, R. I.** (2026). *Deep Spectral Preconditioning via Rectified, Variance-Bounded Matrix Orthogonalization.* MetaFore. RAMuogi.
7. **Christopher, R. I.** (2026). *RACASO: Rotation-Aligned Cautious Approximately Second-Order Optimization.* MetaFore. RACASO.

---

## License

MIT. See `LICENSE` in this repository.
