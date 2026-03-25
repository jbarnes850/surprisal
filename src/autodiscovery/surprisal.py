from dataclasses import dataclass
from scipy.special import betaln, digamma


@dataclass
class SurprisalResult:
    prior_alpha: float
    prior_beta: float
    posterior_alpha: float
    posterior_beta: float
    bayesian_surprise: float
    belief_shifted: bool
    surprisal: int  # binary: 0 or 1


def estimate_beta_params(k: int, n: int) -> tuple[float, float]:
    """Estimate Beta distribution parameters from k successes in n trials.
    Uses uninformed Beta(1,1) prior."""
    return (1 + k, 1 + n - k)


def kl_divergence_beta(alpha2: float, beta2: float,
                        alpha1: float, beta1: float) -> float:
    """D_KL(Beta(alpha2, beta2) || Beta(alpha1, beta1))
    Closed-form KL divergence between two Beta distributions."""
    return float(
        betaln(alpha1, beta1) - betaln(alpha2, beta2)
        + (alpha2 - alpha1) * digamma(alpha2)
        + (beta2 - beta1) * digamma(beta2)
        - (alpha2 - alpha1 + beta2 - beta1) * digamma(alpha2 + beta2)
    )


def bayesian_surprise_with_shift(prior_alpha: float, prior_beta: float,
                                  posterior_alpha: float, posterior_beta: float,
                                  delta: float = 0.5) -> float:
    """BS_shift: KL divergence if belief crosses decision boundary, else 0.
    Implements Equation 4 from AutoDiscovery paper."""
    e_prior = prior_alpha / (prior_alpha + prior_beta)
    e_posterior = posterior_alpha / (posterior_alpha + posterior_beta)
    crosses_boundary = (e_posterior - delta) * (e_prior - delta) <= 0
    beliefs_changed = e_posterior != e_prior
    if crosses_boundary and beliefs_changed:
        return kl_divergence_beta(posterior_alpha, posterior_beta,
                                   prior_alpha, prior_beta)
    return 0.0


def compute_surprisal(k_prior: int, k_post: int, n: int = 30,
                       delta: float = 0.5) -> SurprisalResult:
    """Compute full surprisal result from belief sample counts.
    Uses independent estimation (deliberate deviation from paper)."""
    prior_alpha, prior_beta = estimate_beta_params(k_prior, n)
    posterior_alpha, posterior_beta = estimate_beta_params(k_post, n)
    bs = bayesian_surprise_with_shift(prior_alpha, prior_beta,
                                       posterior_alpha, posterior_beta, delta)
    shifted = bool(bs > 0)
    return SurprisalResult(
        prior_alpha=prior_alpha, prior_beta=prior_beta,
        posterior_alpha=posterior_alpha, posterior_beta=posterior_beta,
        bayesian_surprise=bs, belief_shifted=shifted,
        surprisal=1 if shifted else 0,
    )
