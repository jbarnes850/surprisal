from pathlib import Path
import surprisal


def test_all_prompts_exist():
    prompts_dir = Path(surprisal.__file__).parent / "prompts"
    expected = [
        "experiment_generator.md",
        "experiment_analyst.md", "experiment_reviewer.md",
        "experiment_reviser.md", "experiment_runner.md",
        "hypothesis_generator.md",
        "belief.md", "dedup.md",
    ]
    for name in expected:
        assert (prompts_dir / name).exists(), f"Missing prompt: {name}"


def test_experiment_generator_has_domain_placeholder():
    prompts_dir = Path(surprisal.__file__).parent / "prompts"
    text = (prompts_dir / "experiment_generator.md").read_text()
    assert "{domain}" in text


def test_belief_has_hypothesis_placeholder():
    prompts_dir = Path(surprisal.__file__).parent / "prompts"
    text = (prompts_dir / "belief.md").read_text()
    assert "{hypothesis}" in text
    assert "{evidence_section}" in text


def test_dedup_has_hypothesis_placeholders():
    prompts_dir = Path(surprisal.__file__).parent / "prompts"
    text = (prompts_dir / "dedup.md").read_text()
    assert "{hypothesis_1}" in text
    assert "{hypothesis_2}" in text


def test_experiment_generator_has_literature_section():
    prompts_dir = Path(surprisal.__file__).parent / "prompts"
    text = (prompts_dir / "experiment_generator.md").read_text()
    assert "Literature Grounding" in text
    assert "cited_papers" in text
    assert "arxiv_id" in text


def test_experiment_generator_prefers_real_accessible_resources():
    prompts_dir = Path(surprisal.__file__).parent / "prompts"
    text = (prompts_dir / "experiment_generator.md").read_text()
    assert "HuggingFace datasets" in text
    assert "public model" in text
    assert "Synthetic or simulated data is acceptable only" in text
    assert "Do not artificially downscope" in text


def test_experiment_reviser_does_not_default_to_toy_synthetic_plans():
    prompts_dir = Path(surprisal.__file__).parent / "prompts"
    text = (prompts_dir / "experiment_reviser.md").read_text()
    assert "real HF dataset or public benchmark" in text
    assert "Do not default to synthetic data unless it is justified." in text
    assert "Downgrading every failed plan into a toy synthetic correlation test." in text


def test_analyst_prompt_requires_fidelity_and_validity_checks():
    prompts_dir = Path(surprisal.__file__).parent / "prompts"
    text = (prompts_dir / "experiment_analyst.md").read_text()
    assert "did not actually implement the plan" in text
    assert "silently substituted a toy or unrelated experiment" in text
    assert "Be strict about fidelity and validity." in text


def test_reviewer_prompt_requires_research_grade_evidence():
    prompts_dir = Path(surprisal.__file__).parent / "prompts"
    text = (prompts_dir / "experiment_reviewer.md").read_text()
    assert "Approve only when" in text
    assert "provider or formatting failures prevent reliable interpretation" in text
    assert "Do not approve merely because" in text
