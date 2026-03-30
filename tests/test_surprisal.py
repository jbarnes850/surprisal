import pytest
from surprisal.bayesian import (
    LIKERT_MAP,
    PRIOR_CLAMP_MIN,
    PRIOR_CLAMP_MAX,
    clamp_prior_scores,
    estimate_beta_from_likert,
    kl_divergence_beta,
    bayesian_surprise_kl,
    compute_surprisal,
)


def test_likert_map_has_five_levels():
    assert len(LIKERT_MAP) == 5
    assert LIKERT_MAP["definitely_true"] == 1.0
    assert LIKERT_MAP["definitely_false"] == 0.0
    assert LIKERT_MAP["uncertain"] == 0.5


def test_estimate_beta_from_likert_all_high():
    scores = [1.0] * 10
    alpha, beta = estimate_beta_from_likert(scores)
    # Jeffreys prior 0.5 + 1.0 * 10 = 10.5
    assert alpha == pytest.approx(10.5)
    assert beta == pytest.approx(0.5)


def test_estimate_beta_from_likert_all_low():
    scores = [0.0] * 10
    alpha, beta = estimate_beta_from_likert(scores)
    assert alpha == pytest.approx(0.5)
    assert beta == pytest.approx(10.5)


def test_estimate_beta_from_likert_mixed():
    scores = [0.5] * 10
    alpha, beta = estimate_beta_from_likert(scores)
    # 0.5 + 1.0 * 5.0 = 5.5
    assert alpha == pytest.approx(5.5)
    assert beta == pytest.approx(5.5)


def test_estimate_beta_from_likert_evidence_weight():
    scores = [0.75] * 10
    alpha_ew1, beta_ew1 = estimate_beta_from_likert(scores, evidence_weight=1.0)
    alpha_ew2, beta_ew2 = estimate_beta_from_likert(scores, evidence_weight=2.0)
    # Higher evidence weight produces more concentrated distribution
    assert alpha_ew2 > alpha_ew1
    assert beta_ew2 > beta_ew1


def test_kl_divergence_identical_distributions():
    kl = kl_divergence_beta(10, 10, 10, 10)
    assert kl == pytest.approx(0.0, abs=1e-10)


def test_kl_divergence_different_distributions():
    kl = kl_divergence_beta(9, 23, 23, 9)
    assert kl > 1.0


def test_kl_divergence_is_asymmetric():
    kl_forward = kl_divergence_beta(2, 8, 5, 5)
    kl_backward = kl_divergence_beta(5, 5, 2, 8)
    assert kl_forward != pytest.approx(kl_backward, abs=0.01)


def test_bayesian_surprise_kl_nonzero_for_different_distributions():
    bs = bayesian_surprise_kl(10, 10, 5, 15, kl_scale=5.0)
    assert bs > 0.0


def test_bayesian_surprise_kl_zero_for_identical():
    bs = bayesian_surprise_kl(10, 10, 10, 10, kl_scale=5.0)
    assert bs == pytest.approx(0.0, abs=1e-10)


def test_bayesian_surprise_kl_nonzero_same_side_of_05():
    """KL is nonzero when both means are >0.5 but distributions differ.
    This was the core bug with BS_shift."""
    # Prior: Beta(20, 10) -> mean 0.667
    # Posterior: Beta(25, 5) -> mean 0.833
    bs = bayesian_surprise_kl(20, 10, 25, 5, kl_scale=5.0)
    assert bs > 0.0


def test_compute_surprisal_from_likert_scores():
    prior_scores = [0.75] * 10  # mostly maybe_true
    posterior_scores = [0.25] * 10  # mostly maybe_false
    result = compute_surprisal(prior_scores, posterior_scores)
    assert result.bayesian_surprise > 0
    assert result.kl_raw > 0
    assert result.prior_alpha > 0
    assert result.posterior_alpha > 0


def test_compute_surprisal_identical_scores():
    scores = [0.5] * 10
    result = compute_surprisal(scores, scores)
    # Different evidence weights mean different Beta params even for same scores
    # So KL should be nonzero (ew=1.0 vs ew=2.0)
    assert result.kl_raw > 0


def test_compute_surprisal_evidence_weight_matters():
    prior = [0.75] * 10
    posterior = [0.75] * 10
    r1 = compute_surprisal(prior, posterior, evidence_weight=1.0)
    r2 = compute_surprisal(prior, posterior, evidence_weight=2.0)
    # With ew=1.0 and identical scores, prior and posterior Betas are the same
    assert r1.kl_raw == pytest.approx(0.0, abs=1e-10)
    # With ew=2.0 and identical scores, posterior Beta is more concentrated
    assert r2.kl_raw > 0


def test_clamp_prior_scores_leaves_moderate_scores_unchanged():
    scores = [0.5] * 10
    clamped = clamp_prior_scores(scores)
    assert clamped == scores


def test_clamp_prior_scores_clips_extremes():
    scores = [1.0] * 10  # all definitely_true
    clamped = clamp_prior_scores(scores)
    assert all(s == PRIOR_CLAMP_MAX for s in clamped)


def test_clamp_prior_scores_clips_low_extremes():
    scores = [0.0] * 10  # all definitely_false
    clamped = clamp_prior_scores(scores)
    assert all(s == PRIOR_CLAMP_MIN for s in clamped)


def test_compute_surprisal_overconfident_prior_is_clamped():
    """The bug: prior=[1.0]*10 produced Beta(10.5, 0.5) with mean ~0.95.
    With clamping, prior scores are clipped to 0.9, producing a moderate prior."""
    prior_scores = [1.0] * 10  # overconfident: all definitely_true
    posterior_scores = [0.5] * 10  # uncertain posterior
    result = compute_surprisal(prior_scores, posterior_scores)
    prior_mean = result.prior_alpha / (result.prior_alpha + result.prior_beta)
    # Prior mean should be at most 0.9 after clamping
    assert prior_mean <= 0.91
    # Should still produce nonzero surprise
    assert result.bayesian_surprise > 0
