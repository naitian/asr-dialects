"""
Load gold + system transcriptions into a single tidy DataFrame for analysis.

``load(corpora, models)`` returns one row per (utterance x model) with the chunk
metadata, the gold transcript, and each system's output. Normalization is
applied *at load time* from each corpus's profile (raw text stays canonical on
disk), so revising normalization rules just means reloading.

    df = load(["coraal", "scosya"], ["whisper-large"])
    df.groupby(["corpus", "model", "speaker"])  # ready for analysis

Layout on disk:
    data/chunks/{corpus}_chunks.tsv               -- gold side (see asr.corpora)
    data/transcriptions/{corpus}__{model}.tsv     -- one file per system
"""

import json
from pathlib import Path

import pandas as pd

from asr.normalize import gold_normalizer, system_normalizer
from asr.utils import DATA_DIR

# Tidy column order for the loaded frame.
COLUMNS = [
    "corpus",
    "utterance_id",
    "speaker",
    "line_no",
    "start_time",
    "end_time",
    "gold_text",
    "gold_norm",
    "model",
    "system_text",
    "system_norm",
    "audio_path",
    "utterance_path",
    "transcript_path",
    "extra",
]


def chunks_path(corpus: str) -> Path:
    return DATA_DIR / "chunks" / f"{corpus}_chunks.tsv"


def transcription_path(corpus: str, model: str) -> Path:
    return DATA_DIR / "transcriptions" / f"{corpus}__{model}.tsv"


def load_chunks(corpus: str) -> pd.DataFrame:
    """Load the gold-side chunks for a corpus, parsing the ``extra`` column."""
    df = pd.read_csv(chunks_path(corpus), sep="\t")
    if "extra" in df.columns:
        df["extra"] = df["extra"].apply(
            lambda s: json.loads(s) if isinstance(s, str) and s else {}
        )
    return df


def load_transcriptions(corpus: str, model: str) -> pd.DataFrame:
    """Load one system's output for a corpus as ``utterance_id, system_text, model``."""
    path = transcription_path(corpus, model)
    if not path.exists():
        # Backwards compatibility with the pre-per-model layout.
        legacy = DATA_DIR / "transcriptions" / f"{corpus}.tsv"
        if not legacy.exists():
            raise FileNotFoundError(
                f"No transcriptions for corpus={corpus!r} model={model!r} "
                f"(looked for {path} and {legacy})"
            )
        path = legacy

    df = pd.read_csv(path, sep="\t")
    if "system_text" not in df.columns and "transcription" in df.columns:
        df = df.rename(columns={"transcription": "system_text"})
    df["model"] = model
    return df[["utterance_id", "system_text", "model"]]


def load(
    corpora: list[str],
    models: list[str],
    normalize: bool = True,
) -> pd.DataFrame:
    """Tidy long-format frame: one row per (utterance x model).

    Each corpus's normalization profile is used to compute ``gold_norm`` (markup
    stripped) and ``system_norm`` (markup kept, since model output has none).
    """
    frames: list[pd.DataFrame] = []
    for corpus in corpora:
        chunks = load_chunks(corpus)
        chunks["corpus"] = corpus  # always present, even for legacy chunk files
        if normalize:
            gold_norm = gold_normalizer(corpus)
            chunks["gold_norm"] = (
                chunks["gold_text"].fillna("").astype(str).map(gold_norm)
            )
            system_norm = system_normalizer(corpus)

        for model in models:
            transcriptions = load_transcriptions(corpus, model)
            merged = chunks.merge(transcriptions, on="utterance_id", how="left")
            if normalize:
                merged["system_norm"] = (
                    merged["system_text"].fillna("").astype(str).map(system_norm)
                )
            frames.append(merged)

    result = pd.concat(frames, ignore_index=True)
    ordered = [col for col in COLUMNS if col in result.columns]
    remaining = [col for col in result.columns if col not in ordered]
    return result[ordered + remaining]
