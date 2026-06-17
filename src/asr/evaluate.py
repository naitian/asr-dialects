"""
Evaluate system transcriptions against gold, using normalized text.

    python -m asr.evaluate diagnostic coraal scosya --model whisper-large

Loads the tidy frame from ``asr.dataset.load`` (which applies each corpus's
normalization profile) and reports exact-match and word error rate per
(corpus, model). This replaces the old ``standardize.py``: normalization now
lives in ``asr.normalize`` and is selected per corpus automatically, so there is
no ``type=`` flag to pass.

The per-utterance WER is also written to ``data/wer/{corpus}__{model}.tsv`` so
``asr.dataset.load`` can merge it back in without re-scoring.
"""

import pandas as pd
import typer
from Levenshtein import distance

from asr.dataset import load, save_wer

app = typer.Typer()


def word_error_rate(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return (edit_distance, reference_length) at the word level."""
    ref = reference.split()
    hyp = hypothesis.split()
    edit_distance = distance(ref, hyp)
    return edit_distance, len(ref)


@app.command()
def diagnostic(
    corpora: list[str],
    model: list[str] = typer.Option(["whisper-large"], help="model name(s) to score"),
    normalize: bool = True,
    save: bool = typer.Option(
        True, help="write per-utterance WER to data/wer/{corpus}__{model}.tsv"
    ),
):
    """Report exact-match and WER per (corpus, model)."""
    df = load(corpora, model, normalize=normalize)
    df = df.dropna(subset=["system_text"])

    gold_col = "gold_norm" if normalize else "gold_text"
    system_col = "system_norm" if normalize else "system_text"

    for (corpus, model_name), group in df.groupby(["corpus", "model"]):
        gold = group[gold_col].fillna("").astype(str)
        system = group[system_col].fillna("").astype(str)

        print(f"Evaluating {model_name} outputs for {corpus}")
        exact_match = (gold == system).mean()

        # score only non-empty gold utterances (WER is undefined otherwise)
        keep = gold.str.strip() != ""
        ids = group["utterance_id"][keep].astype(str)
        refs = gold[keep]
        hyps = system[keep]

        records = []
        errors = lengths = 0
        for utterance_id, ref, hyp in zip(ids, refs, hyps):
            dist, ref_len = word_error_rate(ref, hyp)
            errors += dist
            lengths += ref_len
            records.append(
                {
                    "utterance_id": utterance_id,
                    "edit_distance": dist,
                    "ref_length": ref_len,
                    "wer": dist / ref_len if ref_len else float("nan"),
                }
            )
        wer = errors / lengths if lengths else float("nan")

        print(
            f"{corpus} / {model_name}: "
            f"n={len(group)} em={exact_match:.4f} wer={wer:.4f}"
        )

        if save and records:
            # fingerprint covers the scored utterances' (gold, system) pairs so
            # load() can validate the whole file before reusing it.
            out_path = save_wer(
                corpus,
                model_name,
                pd.DataFrame(records),
                gold_norm=refs,
                system_norm=hyps,
            )
            print(f"Wrote {out_path}")


if __name__ == "__main__":
    app()
