"""
Evaluate system transcriptions against gold, using normalized text.

    python -m asr.evaluate diagnostic coraal scosya --model whisper-large

Loads the tidy frame from ``asr.dataset.load`` (which applies each corpus's
normalization profile) and reports exact-match and word error rate per
(corpus, model). This replaces the old ``standardize.py``: normalization now
lives in ``asr.normalize`` and is selected per corpus automatically, so there is
no ``type=`` flag to pass.
"""

import typer

from asr.dataset import load

app = typer.Typer()


def word_error_rate(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return (edit_distance, reference_length) at the word level."""
    ref = reference.split()
    hyp = hypothesis.split()
    # Levenshtein over words
    prev = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        curr = [i] + [0] * len(hyp)
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1], len(ref)


@app.command()
def diagnostic(
    corpora: list[str],
    model: list[str] = typer.Option(["whisper-large"], help="model name(s) to score"),
    normalize: bool = True,
):
    """Report exact-match and WER per (corpus, model)."""
    df = load(corpora, model, normalize=normalize)
    df = df.dropna(subset=["system_text"])

    gold_col = "gold_norm" if normalize else "gold_text"
    system_col = "system_norm" if normalize else "system_text"

    for (corpus, model_name), group in df.groupby(["corpus", "model"]):
        gold = group[gold_col].fillna("").astype(str)
        system = group[system_col].fillna("").astype(str)

        exact_match = (gold == system).mean()
        errors = lengths = 0
        for ref, hyp in zip(gold, system):
            dist, ref_len = word_error_rate(ref, hyp)
            errors += dist
            lengths += ref_len
        wer = errors / lengths if lengths else float("nan")

        print(
            f"{corpus} / {model_name}: "
            f"n={len(group)} em={exact_match:.4f} wer={wer:.4f}"
        )


if __name__ == "__main__":
    app()
