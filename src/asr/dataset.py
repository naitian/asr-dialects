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
    data/norm/{corpus}__gold.tsv (+ .fp)          -- cached gold_norm
    data/norm/{corpus}__{model}__system.tsv (+ .fp)  -- cached system_norm
    data/wer/{corpus}__{model}.tsv (+ .fp)        -- per-utterance WER (optional)

Caches under ``data/norm`` and ``data/wer`` are written here / by ``asr.evaluate``
and reused on later loads; absence is never an error. Each is a whole-corpus
(corpus x model) file, so invalidation is at the *file* level: a sidecar
``.fp`` holds one fingerprint over all rows (see ``normalizer_signature`` /
``wer_fingerprint``), and a rule change in ``asr.normalize`` or any edited
transcript simply rebuilds the file -- no manual cache busting needed.
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
    "region",
    "start_time",
    "end_time",
    "gold_text",
    "gold_norm",
    "model",
    "system_text",
    "system_norm",
    "edit_distance",
    "ref_length",
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
#
# Every cache is a whole corpus-level (corpus x model) file, so we fingerprint
# at the *file* level rather than per row: one hash over all rows decides whether
# the file is reusable. The fingerprint lives in a sidecar ``<cache>.fp`` next to
# the ``.tsv``. It is order-independent (rows are sorted by ``utterance_id``
# before hashing) so it survives incidental reordering.
# --------------------------------------------------------------------------- #
def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _file_fingerprint(prefix: str, ids: pd.Series, *values: pd.Series) -> str:
    """Order-independent hash of ``(utterance_id, *values)`` rows, salted by ``prefix``."""
    frame = pd.DataFrame({"id": ids.astype(str).to_numpy()})
    for index, column in enumerate(values):
        frame[index] = column.fillna("").astype(str).to_numpy()
    frame = frame.sort_values("id", kind="stable")
    hasher = hashlib.sha1()
    hasher.update(prefix.encode("utf-8"))
    for column in frame.columns:
        hasher.update(b"\x01")
        hasher.update("\x00".join(frame[column]).encode("utf-8"))
    return hasher.hexdigest()


def _fingerprint_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".fp")


def _read_fingerprint(cache_path: Path) -> str | None:
    path = _fingerprint_path(cache_path)
    return path.read_text().strip() if path.exists() else None


def normalizer_signature() -> str:
    """Fingerprint of the normalization *rules*, salted into the norm cache.

    A cached normalization can't be checked without re-running the (slow)
    normalizer, so the file fingerprint folds in everything that determines the
    rules: the source of ``asr.normalize`` plus the whisper version (the whisper
    normalizers carry their own lexicons). Any edit there changes the signature
    and forces a full recompute on the next load.
    """
    import whisper

    version = getattr(whisper, "__version__", "?")
    source = Path(normalize_module.__file__).read_bytes()
    return _sha1(source.decode("utf-8", "replace") + "\x00" + version)


def wer_fingerprint(
    utterance_id: pd.Series, gold_norm: pd.Series, system_norm: pd.Series
) -> str:
    """File-level fingerprint of a WER cache: one hash over its ``(id, gold, system)``.

    WER is a pure function of the normalized pair, so the cache stays valid iff
    every cached utterance's ``(gold_norm, system_norm)`` is unchanged --
    independent of *which* rules produced them, hence no rule signature here.
    ``load`` recomputes this over the same utterance set and compares.
    """
    return _file_fingerprint("", utterance_id, gold_norm, system_norm)


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

    # keep_default_na=False so an empty transcription stays "" (a real, scorable
    # empty output) and a clip transcribed as a NA-like token (e.g. "null") keeps
    # its literal text. After the merge in ``load``, only a genuinely missing
    # system row becomes NaN, which is what ``evaluate`` drops on.
    df = pd.read_csv(path, sep="\t", keep_default_na=False)
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


def save_wer(
    corpus: str,
    model: str,
    records: pd.DataFrame,
    *,
    gold_norm: pd.Series,
    system_norm: pd.Series,
) -> Path:
    """Write the per-utterance WER cache plus its fingerprint sidecar.

    ``records`` holds the scored rows (``utterance_id, edit_distance, ref_length,
    wer``); ``gold_norm``/``system_norm`` are the normalized columns those scores
    came from, positionally aligned with ``records``. The file fingerprint is
    taken over that same utterance set so ``load`` can validate it.
    """
    path = wer_path(corpus, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(path, sep="\t", index=False)
    fingerprint = wer_fingerprint(records["utterance_id"], gold_norm, system_norm)
    _fingerprint_path(path).write_text(fingerprint)
    return path


def cached_normalize(
    df: pd.DataFrame,
    *,
    text_col: str,
    normalizer: Normalizer,
    cache_path: Path,
    signature: str,
    use_cache: bool = True,
) -> pd.Series:
    """Normalize ``df[text_col]``, reusing the whole ``cache_path`` when it's current.

    The cache is validated at the file level: if its sidecar fingerprint matches
    ``hash(signature + all inputs)``, the stored norms are mapped straight back by
    ``utterance_id``; otherwise the normalizer is re-run over every row and the
    cache (plus fingerprint) is rewritten. Returns a Series aligned to ``df.index``.
    """
    ids = df["utterance_id"].astype(str)
    inputs = df[text_col].fillna("").astype(str)
    fingerprint = _file_fingerprint(signature, ids, inputs)

    if use_cache and cache_path.exists() and _read_fingerprint(cache_path) == fingerprint:
        cached = pd.read_csv(cache_path, sep="\t", dtype=str, keep_default_na=False)
        norm_by_id = dict(zip(cached["utterance_id"], cached["norm"]))
        print(f"  {cache_path.name}: cache hit ({len(df)} rows)")
        return ids.map(norm_by_id)

    values = [normalizer(text) for text in inputs]
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"utterance_id": ids.to_numpy(), "norm": values}).to_csv(
            cache_path, sep="\t", index=False
        )
        _fingerprint_path(cache_path).write_text(fingerprint)
    print(f"  {cache_path.name}: recomputed ({len(df)} rows)")
    return pd.Series(values, index=df.index)


def _attach_cached_wer(
    merged: pd.DataFrame, saved_wer: pd.DataFrame, cache_path: Path
) -> pd.DataFrame:
    """Left-merge the cached WER columns iff the WER file fingerprint is current.

    The fingerprint is computed over the cache's own utterance set (WER omits
    empty-gold rows), so we restrict ``merged`` to those IDs before re-hashing
    its ``gold_norm``/``system_norm``. A mismatch -- or a fingerprint-less legacy
    file -- means the whole cache is stale and is simply not merged.
    """
    cached_ids = set(saved_wer["utterance_id"].astype(str))
    subset = merged[merged["utterance_id"].astype(str).isin(cached_ids)]
    current = wer_fingerprint(
        subset["utterance_id"], subset["gold_norm"], subset["system_norm"]
    )
    if _read_fingerprint(cache_path) != current:
        return merged
    return merged.merge(
        saved_wer[["utterance_id", "wer", "edit_distance", "ref_length"]],
        on="utterance_id",
        how="left",
    )


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
                    merged = _attach_cached_wer(
                        merged, saved_wer, wer_path(corpus, model)
                    )
            frames.append(merged)

    result = pd.concat(frames, ignore_index=True)
    ordered = [col for col in COLUMNS if col in result.columns]
    remaining = [col for col in result.columns if col not in ordered]
    return result[ordered + remaining]
