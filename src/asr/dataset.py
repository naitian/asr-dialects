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
    data/norm/{corpus}__gold.tsv                  -- cached gold_norm
    data/norm/{corpus}__{model}__system.tsv       -- cached system_norm
    data/wer/{corpus}__{model}.tsv                -- per-utterance WER (optional)

Caches under ``data/norm`` and ``data/wer`` are written here / by ``asr.evaluate``
and reused on later loads; absence is never an error. Both self-invalidate per
row via a fingerprint (see ``normalizer_signature`` / ``wer_fingerprint``), so a
rule change in ``asr.normalize`` or an edited transcript recomputes only the
affected rows -- no manual cache busting needed.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

from asr import normalize as normalize_module
from asr.normalize import Normalizer, gold_normalizer, system_normalizer
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
    "wer",
    "audio_path",
    "utterance_path",
    "transcript_path",
    "extra",
]


def chunks_path(corpus: str) -> Path:
    return DATA_DIR / "chunks" / f"{corpus}_chunks.tsv"


def transcription_path(corpus: str, model: str) -> Path:
    return DATA_DIR / "transcriptions" / f"{corpus}__{model}.tsv"


def wer_path(corpus: str, model: str) -> Path:
    return DATA_DIR / "wer" / f"{corpus}__{model}.tsv"


def norm_cache_path(corpus: str, key: str) -> Path:
    """``key`` is ``"gold"`` or ``"{model}__system"``."""
    return DATA_DIR / "norm" / f"{corpus}__{key}.tsv"


# --------------------------------------------------------------------------- #
# Cache fingerprints (invalidation)
# --------------------------------------------------------------------------- #
def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalizer_signature() -> str:
    """Fingerprint of the normalization *rules*, for the norm cache.

    A cached normalization can't be checked without re-running the (slow)
    normalizer, so we instead fingerprint each row as ``hash(signature + input)``
    where the signature covers everything that determines the rules: the source
    of ``asr.normalize`` plus the whisper version (the whisper normalizers carry
    their own lexicons). Any edit there changes the signature and forces a
    recompute on the next load.
    """
    import whisper

    version = getattr(whisper, "__version__", "?")
    source = Path(normalize_module.__file__).read_bytes()
    return _sha1(source.decode("utf-8", "replace") + "\x00" + version)


def wer_fingerprint(gold_norm: pd.Series, system_norm: pd.Series) -> pd.Series:
    """Per-row fingerprint of the normalized (gold, system) pair, for the WER cache.

    WER is a pure function of these two strings, so a cached WER stays valid iff
    they are unchanged -- independent of *which* rules produced them. This is why
    the WER fingerprint needs no rule signature: ``load`` recomputes the (now
    cached, cheap) normalizations and compares.
    """
    gold = gold_norm.fillna("").astype(str)
    system = system_norm.fillna("").astype(str)
    return (gold + "\x00" + system).map(_sha1)


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


def load_wer(corpus: str, model: str) -> pd.DataFrame | None:
    """Load saved per-utterance WER for a corpus/model, or ``None`` if absent.

    Written by ``asr.evaluate``; ``utterance_id, wer`` are the columns merged
    back into the tidy frame.
    """
    path = wer_path(corpus, model)
    if not path.exists():
        return None
    return pd.read_csv(path, sep="\t")


def cached_normalize(
    df: pd.DataFrame,
    *,
    text_col: str,
    normalizer: Normalizer,
    cache_path: Path,
    signature: str,
    use_cache: bool = True,
) -> pd.Series:
    """Normalize ``df[text_col]``, reusing ``cache_path`` for unchanged rows.

    A row is reused iff ``hash(signature + input_text)`` matches the cached
    fingerprint; otherwise the normalizer is (re)run for that row only. The
    refreshed cache (covering exactly the rows just seen) is written back, so
    stale/removed utterances drop out on their own. Returns a Series aligned to
    ``df.index``.
    """
    inputs = df[text_col].fillna("").astype(str)
    ids = df["utterance_id"].astype(str)
    fingerprints = (signature + "\x00" + inputs).map(_sha1)

    cached: dict[str, tuple[str, str]] = {}
    if use_cache and cache_path.exists():
        prev = pd.read_csv(cache_path, sep="\t", dtype=str, keep_default_na=False)
        cached = {
            uid: (fp, norm)
            for uid, fp, norm in zip(prev["utterance_id"], prev["fingerprint"], prev["norm"])
        }

    values: list[str] = []
    hits = 0
    for uid, text, fingerprint in zip(ids, inputs, fingerprints):
        entry = cached.get(uid)
        if entry is not None and entry[0] == fingerprint:
            values.append(entry[1])
            hits += 1
        else:
            values.append(normalizer(text))

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"utterance_id": ids.values, "fingerprint": fingerprints.values, "norm": values}
        ).to_csv(cache_path, sep="\t", index=False)

    print(f"  {cache_path.name}: {hits}/{len(values)} cached, {len(values) - hits} recomputed")
    return pd.Series(values, index=df.index)


def _attach_cached_wer(merged: pd.DataFrame, saved_wer: pd.DataFrame) -> pd.DataFrame:
    """Left-merge the cached ``wer`` column, dropping rows whose fingerprint is stale.

    Requires ``gold_norm``/``system_norm`` on ``merged`` to recompute the current
    fingerprint. Files predating the fingerprint column are trusted as-is.
    """
    has_fp = "fingerprint" in saved_wer.columns
    cols = ["utterance_id", "wer", "edit_distance", "ref_length"] + (["fingerprint"] if has_fp else [])
    merged = merged.merge(saved_wer[cols], on="utterance_id", how="left")
    if has_fp:
        current = wer_fingerprint(merged["gold_norm"], merged["system_norm"])
        stale = merged["fingerprint"].notna() & (merged["fingerprint"] != current)
        merged.loc[stale, "wer"] = pd.NA
        merged = merged.drop(columns="fingerprint")
    return merged


def load(
    corpora: list[str],
    models: list[str],
    normalize: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Tidy long-format frame: one row per (utterance x model).

    Each corpus's normalization profile is used to compute ``gold_norm`` (markup
    stripped) and ``system_norm`` (markup kept, since model output has none).
    Normalization is the slow step, so results are cached under ``data/norm`` and
    reused for rows whose input and the normalization rules are unchanged; the
    saved per-utterance WER is likewise merged in when its fingerprint still
    matches. Pass ``use_cache=False`` to ignore (and not write) the norm cache.
    """
    print("Loading corpora...")
    signature = normalizer_signature() if normalize else ""
    frames: list[pd.DataFrame] = []
    for corpus in corpora:
        chunks = load_chunks(corpus)
        chunks["corpus"] = corpus  # always present, even for legacy chunk files
        if normalize:
            chunks["gold_norm"] = cached_normalize(
                chunks,
                text_col="gold_text",
                normalizer=gold_normalizer(corpus),
                cache_path=norm_cache_path(corpus, "gold"),
                signature=signature,
                use_cache=use_cache,
            )
            system_norm = system_normalizer(corpus)

        for model in models:
            transcriptions = load_transcriptions(corpus, model)
            merged = chunks.merge(transcriptions, on="utterance_id", how="left")
            if normalize:
                merged["system_norm"] = cached_normalize(
                    merged,
                    text_col="system_text",
                    normalizer=system_norm,
                    cache_path=norm_cache_path(corpus, f"{model}__system"),
                    signature=signature,
                    use_cache=use_cache,
                )
                # WER is validated against gold_norm/system_norm, so only merge
                # it on the normalized path.
                saved_wer = load_wer(corpus, model)
                if saved_wer is not None:
                    merged = _attach_cached_wer(merged, saved_wer)
            frames.append(merged)

    result = pd.concat(frames, ignore_index=True)
    ordered = [col for col in COLUMNS if col in result.columns]
    remaining = [col for col in result.columns if col not in ordered]
    return result[ordered + remaining]
