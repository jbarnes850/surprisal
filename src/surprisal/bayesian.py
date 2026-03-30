import logging
from dataclasses import dataclass
from scipy.special import betaln, digamma

logger = logging.getLogger("surprisal")

LIKERT_MAP: dict[str, float] = {
    "definitely_true": 1.0,
    "maybe_true": 0.75,
    "uncertain": 0.5,
    "maybe_false": 0.25,
    "definitely_false": 0.0,
}

# Prior probability bounds — no hypothesis starts as "certain"
PRIOR_CLAMP_MIN = 0.1
PRIOR_CLAMP_MAX = 0.9


@dataclass
class SurprisalResult:
    prior_alpha: float
    prior_beta: float
    posterior_alpha: float
    posterior_beta: float
    bayesian_surprise: float
    kl_raw: float


def clamp_prior_scores(scores: list[float],
                       clamp_min: float = PRIOR_CLAMP_MIN,
                       clamp_max: float = PRIOR_CLAMP_MAX) -> list[float]:
    """Clamp prior belief scores away from extremes.
    Prevents degenerate Beta distributions from overconfident priors."""
    clamped = [max(clamp_min, min(clamp_max, s)) for s in scores]
    raw_mean = sum(scores) / len(scores) if scores else 0.5
    clamped_mean = sum(clamped) / len(clamped) if clamped else 0.5
    if abs(raw_mean - clamped_mean) > 0.05:
        logger.warning(
            f"Prior clamped: raw_mean={raw_mean:.3f} → clamped_mean={clamped_mean:.3f}. "
            f"Belief agent may be overconfident."
        )
    return clamped


def estimate_beta_from_likert(scores: list[float],
                               evidence_weight: float = 1.0) -> tuple[float, float]:
    """Estimate Beta parameters from Likert scores (0.0-1.0).
    Uses Jeffreys prior (0.5, 0.5). Evidence weight scales the
    contribution of each observation."""
    alpha = 0.5 + evidence_weight * sum(scores)
    beta = 0.5 + evidence_weight * sum(1.0 - s for s in scores)
    return (alpha, beta)


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


def bayesian_surprise_kl(prior_alpha: float, prior_beta: float,
                          posterior_alpha: float, posterior_beta: float,
                          kl_scale: float = 5.0) -> float:
    """Raw KL(posterior || prior) / kl_scale. No threshold.
    Any distributional shift produces nonzero surprise."""
    kl = kl_divergence_beta(posterior_alpha, posterior_beta,
                             prior_alpha, prior_beta)
    return kl / kl_scale


def compute_surprisal(prior_scores: list[float],
                       posterior_scores: list[float],
                       evidence_weight: float = 2.0,
                       kl_scale: float = 5.0) -> SurprisalResult:
    """Compute surprisal from Likert belief scores.
    Prior scores are clamped to [0.1, 0.9] to prevent degenerate distributions.
    Posterior scores use evidence_weight."""
    clamped_prior = clamp_prior_scores(prior_scores)
    prior_alpha, prior_beta = estimate_beta_from_likert(clamped_prior)
    posterior_alpha, posterior_beta = estimate_beta_from_likert(
        posterior_scores, evidence_weight,
    )
    # Calibration diagnostic
    prior_mean = prior_alpha / (prior_alpha + prior_beta)
    if prior_mean > 0.9 or prior_mean < 0.1:
        logger.warning(
            f"Prior mean {prior_mean:.3f} is extreme despite clamping "
            f"(alpha={prior_alpha:.2f}, beta={prior_beta:.2f}). "
            f"Check belief agent calibration."
        )
    kl_raw = kl_divergence_beta(posterior_alpha, posterior_beta,
                                 prior_alpha, prior_beta)
    bs = kl_raw / kl_scale
    return SurprisalResult(
        prior_alpha=prior_alpha, prior_beta=prior_beta,
        posterior_alpha=posterior_alpha, posterior_beta=posterior_beta,
        bayesian_surprise=bs, kl_raw=kl_raw,
    )
