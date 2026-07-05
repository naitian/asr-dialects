"""
Data model for ASR evaluation.

A :class:`Chunk` is one utterance carved out of a corpus recording (gold side).
A :class:`TranscribedChunk` adds a single system's output for that utterance.

Both are flat dataclasses so they round-trip cleanly to/from the TSV files under
``data/`` and land as real columns when loaded into a DataFrame for analysis.
Corpus-specific fields that don't deserve a top-level column (e.g. CORAAL
socioeconomic/age group) go in ``extra``.
"""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    # identity / provenance
    corpus: str
    utterance_id: str  # unique within the corpus
    speaker: str
    line_no: str
    region: str  # geographic region; shared across corpora
    # timing within the source recording
    start_time: float
    end_time: float
    # the reference transcript for this utterance
    gold_text: str
    # paths
    audio_path: str  # source recording the chunk was cut from
    utterance_path: str  # the extracted per-utterance audio clip
    transcript_path: str  # source transcript file
    # corpus-specific metadata that doesn't warrant its own column
    extra: dict = field(default_factory=dict)


@dataclass
class TranscribedChunk(Chunk):
    system_text: str = ""
    system: str = ""  # name of the model that produced system_text
