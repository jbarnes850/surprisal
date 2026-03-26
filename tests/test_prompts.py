from pathlib import Path
import surprisal


def test_all_prompts_exist():
    prompts_dir = Path(surprisal.__file__).parent / "prompts"
    expected = [
        "experiment_generator.md", "experiment_programmer.md",
        "experiment_analyst.md", "experiment_reviewer.md",
        "experiment_reviser.md", "hypothesis_generator.md",
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
