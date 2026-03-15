from pathlib import Path

from core_llm.data.sft_quality import lint_sft_seed


def test_qa_seed_quality_checks_pass():
    repo_root = Path(__file__).resolve().parents[2]
    qa_seed_path = repo_root / "core_llm" / "data" / "raw" / "sft" / "qa_seed.jsonl"
    issues, category_counts = lint_sft_seed(qa_seed_path)

    assert issues == []
    assert category_counts["definition"] >= 150
    assert category_counts["comparison"] >= 20
    assert category_counts["procedure"] >= 15
