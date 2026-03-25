import pytest
from autodiscovery.surprisal import (
    estimate_beta_params,
    kl_divergence_beta,
    bayesian_surprise_with_shift,
    compute_surprisal,
)


def test_estimate_beta_params_all_true():
    alpha, beta = estimate_beta_params(k=30, n=30)
    assert alpha == 31
    assert beta == 1


def test_estimate_beta_params_all_false():
    alpha, beta = estimate_beta_params(k=0, n=30)
    assert alpha == 1
    assert beta == 31


def test_estimate_beta_params_half():
    alpha, beta = estimate_beta_params(k=15, n=30)
    assert alpha == 16
    assert beta == 16


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


def test_bayesian_surprise_with_shift_crosses_boundary():
    bs = bayesian_surprise_with_shift(23, 9, 9, 23, delta=0.5)
    assert bs > 0.0


def test_bayesian_surprise_with_shift_same_side():
    bs = bayesian_surprise_with_shift(23, 9, 25, 7, delta=0.5)
    assert bs == 0.0


def test_bayesian_surprise_no_change():
    bs = bayesian_surprise_with_shift(15, 15, 15, 15, delta=0.5)
    assert bs == 0.0


def test_compute_surprisal_from_samples():
    result = compute_surprisal(k_prior=22, k_post=8, n=30, delta=0.5)
    assert result.prior_alpha == 23
    assert result.prior_beta == 9
    assert result.posterior_alpha == 9
    assert result.posterior_beta == 23
    assert result.belief_shifted is True
    assert result.surprisal == 1
    assert result.bayesian_surprise > 0


def test_compute_surprisal_no_shift():
    result = compute_surprisal(k_prior=22, k_post=20, n=30, delta=0.5)
    assert result.belief_shifted is False
    assert result.surprisal == 0
