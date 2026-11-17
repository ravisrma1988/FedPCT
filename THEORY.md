# Theory: per-site coverage of personalised federated conformal prediction under drift

Status: 2026-11-03. Replaces `theory_sketch.md` (which treated the selective-labelling variant; the
frozen method uses uniform labelling). Proofs written out in full; constants checked numerically
against the real streams (`src/theory_check.py`, section 5).

## 1. Setting (one site; the site index is dropped)

- Rounds m = 1, ..., T are the site's camera sequences in time order. Round m has n_m >= 0
  known-class images; N = sum_m n_m; n_max = max_m n_m.
- **Data are arbitrary.** The image stream (images, true labels, sequence lengths) is any fixed
  sequence: no exchangeability, no stationarity, no distributional assumption. ("Oblivious": it does
  not react to the site's random labelling decisions. Distribution drift of any form is allowed.)
- **The model may change at every round.** The head in force at round m, f_m, is any function of
  the data and of the feedback that has *arrived* before round m (e.g. FedAvg head, Ditto refits).
- Non-conformity scores s_{m,i} = S(f_m; x_{m,i}, y_{m,i}) in [0, B]. For randomised RAPS,
  B = 1 + lambda_RAPS * max(0, C - k_reg), fixed before deployment.
- Threshold tau_m; the prediction set of image i contains class c iff S(f_m; x_{m,i}, c) <= tau_m.
  Misses M_m = sum_i 1[s_{m,i} > tau_m]; excess e_m = M_m - alpha n_m, so -alpha n_m <= e_m <= (1-alpha) n_m.
- **Feedback.** Round m is labelled with L_m ~ Bernoulli(r), independently of everything else
  (uniform labelling at rate r). Its labels arrive after a delay of D rounds.
- **Update (per-site quantile tracking, the frozen method):**
  tau_{m+1} = tau_m + eta * L_{m-D} * e_{m-D}   (no update while m - D < 1),
  where e_{m-D} is evaluated at the threshold tau_{m-D} used when round m - D was predicted.
  tau_1 = tau_0 in [0, B] (pooled conformal quantile on cal).

## 2. Results

**Lemma 1 (bounded thresholds).** For every round,
  -(D+1) eta alpha n_max  <=  tau_m  <=  B + (D+1) eta (1-alpha) n_max.

*Proof.* Upper: if tau_m >= B, no score exceeds tau_m, so e_m = -alpha n_m <= 0. A positive increment
at step t comes from round t - D, and requires tau_{t-D} < B. Let tau_m > B and let s < m be the last
round with tau_s <= B. Every increment applied after round s comes from a round m' = t - D. Those with
m' > s have tau_{m'} > B and add <= 0. Those with m' <= s arrive at t = m' + D <= s + D: at most
D + 1 of them (t = s, ..., s + D), each adding at most eta (1-alpha) n_max. Hence
tau_m <= tau_s + (D+1) eta (1-alpha) n_max <= B + (D+1) eta (1-alpha) n_max.
Lower: symmetric. If tau_m < 0, every image is missed (scores >= 0), so e_m = (1-alpha) n_m >= 0.
Negative increments come only from rounds with tau >= 0, at most D + 1 of them after the last
non-negative threshold, each >= -eta alpha n_max. []

Write W = B + (D+1) eta n_max (the width of that interval).

**Theorem 1 (long-run per-site coverage for any data and any model sequence).** For every data
stream and every model sequence (f_m) adapted to the arrived feedback, with probability at least
1 - delta over the labelling draws,

  | (1/N) sum_{m<=T} M_m - alpha |  <=  ( W/eta + sqrt(2 log(2/delta) sum_{m<=T-D} n_m^2) ) / (r N)
                                        + (1/N) sum_{T-D < m <= T} n_m.

With r = 1 (every sequence labelled) the middle term vanishes and the bound holds deterministically.

*Proof.* (i) Telescoping. The updates applied up to the end of the stream are the arrivals of rounds
1, ..., T - D, so tau_{T+1} - tau_1 = eta sum_{m <= T-D} L_m e_m. By Lemma 1 both thresholds lie in an
interval of width W, so |sum_{m<=T-D} L_m e_m| <= W / eta.
(ii) Unlabelled rounds. Write sum_{m<=T-D} e_m = (1/r) sum L_m e_m - (1/r) sum Z_m with Z_m = (L_m - r) e_m.
Let G_m be the sigma-field of all labelling draws before round m. Because the data are fixed and the
model f_m and threshold tau_m depend only on arrived feedback (a function of L_1, ..., L_{m-D-1}),
e_m is G_m-measurable, while L_m is independent of G_m with mean r. So E[Z_m | G_m] = 0: (Z_m) is a
martingale difference sequence, with |Z_m| <= max(r, 1-r) n_m <= n_m, and the n_m are fixed numbers.
Azuma-Hoeffding: with probability >= 1 - delta, |sum_{m<=T-D} Z_m| <= sqrt(2 log(2/delta) sum n_m^2).
(iii) The last D rounds have no feedback yet: |sum_{T-D<m<=T} e_m| <= sum_{T-D<m<=T} n_m.
Combine (i)-(iii) and divide by N, using sum_m M_m - alpha N = sum_m e_m. []

**Corollary 1 (personalisation cannot break validity).** Theorem 1 places no condition on (f_m)
beyond using only arrived feedback. So any personalisation rule (Ditto with any lambda, fine-tuning,
local-only) keeps the same long-run per-site coverage guarantee; the model sequence affects only
*efficiency* (set size), never validity. This is the formal reason the method can personalise
aggressively. With K sites, a union bound (delta -> delta / K) gives the guarantee for all sites at once.

**Proposition 1 (a fixed threshold has no such guarantee).** Let tau be any threshold fixed before the
stream (e.g. the pooled conformal quantile at alpha computed on calibration data, with the global or
a personalised head). For every eps > 0 there is a stream with per-site miscoverage >= 1 - eps.
Even with exchangeable data the split-conformal guarantee of tau holds only for the calibrated model
f; once f_m != f (personalisation), the score distribution changes and the coverage of tau is
unconstrained.
*Proof.* Take any stream whose true-label scores all exceed tau (e.g. a species-mix shift towards
classes the head ranks low; under RAPS the scores of low-ranked classes exceed any tau < B). Then every
image is missed. For the second claim, take exchangeable data and a refit f_m that lowers the true
class's probability: the true-label scores increase, while tau was calibrated on f. []

## 3. What the theory predicts, and how it matches the experiments

The bound's leading term is W / (eta r N), plus a noise term of order n_max / (r sqrt(T)), plus
the D trailing rounds. So the guarantee tightens with more labels (r), less delay (D) and longer
streams (N), and it does not depend on the drift at all, while the error of a fixed threshold grows
with the drift (Proposition 1).

| Prediction | Observation (all pre-registered or development runs) |
|---|---|
| Tracking helps more as r grows | Tracking vs matched fixed threshold: null/harmful at r = 0.1 on both datasets and backbones; clearest at r = 1.0 (CCT lockbox -10.4; DINOv2 -4.5 to -6.5) |
| Tracking helps more with short delay | Development delay sweep: -5.5 (D = 0), -5.7 (D = 2), -4.3 (D = 5), -0.8 (D = 10) |
| Fixed thresholds fail more under strong drift; tracking does not depend on drift | Static FCP: 26% sites below on iWildCam, 47% on CCT; the method reaches 8% on both |
| Personalisation does not affect validity | Ditto, fine-tune and local-only heads all keep the tracking behaviour; they differ in set size and in the unknown flag (development ablations) |
| The fixed unknown-flag threshold has no guarantee under drift | Unknown recall met on iWildCam, missed on CCT (0.73-0.77) with both backbones |

## 4. What the theory does not cover (stated in the paper)

- The guarantee is a long-run average over each site's stream. Our endpoint (sites *significantly
  below* 0.9 over the test period) is related but not identical; Theorem 1 does not by itself bound it
  for short streams (section 5 gives the numbers).
- No efficiency (set-size) result: why federated Ditto gives smaller sets than local-only heads is
  empirical. No federated theorem is claimed.
- The unknown flag uses a fixed threshold, so Proposition 1 applies to it: no guarantee under drift.
  A tracked unknown threshold would inherit Theorem 1 (it is the same update on the novelty score) but
  is not part of the frozen method.

## 5. Numerical check on the real streams

See `results/theory_check.json` and the summary below (filled in by `src/theory_check.py`).
