"""
Text normalization for ASR evaluation.

A normalizer is an ordered list of ``Step`` functions (``str -> str``). We build
one from a :class:`NormProfile`, which captures everything that varies *per
corpus*:

    - ``markup_steps``: strips that corpus's annotation syntax (brackets,
      partial words, parenthetical notes, ...). These conventions only exist in
      the hand-transcribed *gold* text, so they are applied to gold only.
    - ``lexicon``: lexical normalizations (contractions, reduced/dialect forms)
      that either side can produce, so they are applied to *both* gold and
      system output. This accounts for variance in how a system transcribes a
      given utterance.

Everything else -- the shared English lexicon and the numeric/spelling/symbol
cleanup tail -- applies to both gold and system. So the *only* difference
between normalizing gold and system text is whether markup is stripped, which
``build(profile, strip_markup=...)`` exposes as a single flag.

We use the whisper normalization as a base, adapted to the SCOSYA and CORAAL
transcription schemes.

NOTE: This might overfit to whisper, so we may want to re-check it when adding a
new system.
"""

import re
from dataclasses import dataclass, field
from typing import Callable

from whisper.normalizers.basic import remove_symbols_and_diacritics
from whisper.normalizers.english import (
    EnglishNumberNormalizer,
    EnglishSpellingNormalizer,
)

# A normalization step transforms a string into a (more normalized) string.
Step = Callable[[str], str]


@dataclass
class Normalizer:
    """An ordered pipeline of steps applied left-to-right."""

    steps: list[Step]

    def __call__(self, text: str) -> str:
        for step in self.steps:
            text = step(text)
        return text


# --------------------------------------------------------------------------- #
# Step builders
# --------------------------------------------------------------------------- #
def regex_sub(pattern: str, replacement: str = "") -> Step:
    """A step that applies a single (precompiled) regex substitution."""
    compiled = re.compile(pattern)
    return lambda text: compiled.sub(replacement, text)


def replace_lexicon(mapping: dict[str, str]) -> Step:
    """A step that applies each ``pattern -> replacement`` in order."""
    compiled = [(re.compile(pattern), repl) for pattern, repl in mapping.items()]

    def step(text: str) -> str:
        for pattern, repl in compiled:
            text = pattern.sub(repl, text)
        return text

    return step


def apply_callable(fn: Callable[[str], str]) -> Step:
    """Wrap a stateful callable (e.g. a whisper normalizer) as a step."""
    return lambda text: fn(text)


# --------------------------------------------------------------------------- #
# Shared lexicon: universal English forms, applied to gold *and* system.
# --------------------------------------------------------------------------- #
SHARED_LEXICON: dict[str, str] = {
    # common contractions
    r"\bwon't\b": "will not",
    r"\bcan't\b": "can not",
    r"\blet's\b": "let us",
    r"\bain't\b": "aint",
    r"\by'all\b": "you all",
    r"\bwanna\b": "want to",
    r"\bgotta\b": "got to",
    r"\bgonna\b": "going to",
    r"\bi'ma\b": "i am going to",
    r"\bimma\b": "i am going to",
    r"\bwoulda\b": "would have",
    r"\bcoulda\b": "could have",
    r"\bshoulda\b": "should have",
    r"\bma'am\b": "madam",
    # contractions in titles/prefixes
    r"\bmr\b": "mister ",
    r"\bmrs\b": "missus ",
    r"\bst\b": "saint ",
    r"\bdr\b": "doctor ",
    r"\bprof\b": "professor ",
    r"\bcapt\b": "captain ",
    r"\bgov\b": "governor ",
    r"\bald\b": "alderman ",
    r"\bgen\b": "general ",
    r"\bsen\b": "senator ",
    r"\brep\b": "representative ",
    r"\bpres\b": "president ",
    r"\brev\b": "reverend ",
    r"\bhon\b": "honorable ",
    r"\basst\b": "assistant ",
    r"\bassoc\b": "associate ",
    r"\blt\b": "lieutenant ",
    r"\bcol\b": "colonel ",
    r"\bjr\b": "junior ",
    r"\bsr\b": "senior ",
    r"\besq\b": "esquire ",
    # perfect tenses, ideally any past participle, but that's harder..
    r"'d been\b": " had been",
    r"'s been\b": " has been",
    r"'dgone\b": " had gone",
    r"'s gone\b": " has gone",
    r"'d done\b": " had done",  # "'s done" is ambiguous
    r"'s got\b": " has got",
    # general contractions
    r"n't\b": " not",
    r"'re\b": " are",
    r"'s\b": " is",
    r"'d\b": " would",
    r"'ll\b": " will",
    r"'t\b": " not",
    r"'ve\b": " have",
    r"'m\b": " am",
    # reduced forms (general informal English; any system may emit these)
    r"\bmusta\b": "must have",
    r"\bmighta\b": "might have",
    r"\bhafta\b": "have to",
    r"\btryna\b": "trying to",
    r"\bsposta\b": "supposed to",
    r"\bfinna\b": "fixing to",  # NOTE: check how this is handled?
    r"\boughta\b": "ought to",
    r"\bcause\b": "because",  # NOTE: defaulting to the CORAAL definition
    r"\btil\b": "until",
    r"\b'em\b": "them",
    r"\blemme\b": "let me",
    r"\bi'm'a\b": "i am going to",
    r"\bwhatchu\b": "what are you",  # could also be "what do you"
    r"\bwhatcha\b": "what are you",
    r"\bgotcha\b": "got you",
}


# --------------------------------------------------------------------------- #
# Numeric tail: number/spelling/symbol cleanup that runs *after* the lexicon.
# Applied to gold *and* system. Built lazily so importing this module doesn't
# construct the whisper normalizers until a pipeline is actually built.
# --------------------------------------------------------------------------- #
IGNORE_FILLERS = r"\b(hmm|mm|mhm|mmm|uh|um|ah)\b"  # "ah" added from CORAAL


def numeric_tail() -> list[Step]:
    standardize_numbers = EnglishNumberNormalizer()
    standardize_spellings = EnglishSpellingNormalizer()
    return [
        regex_sub(r"(\d),(\d)", r"\1\2"),  # commas between digits
        regex_sub(r"\.([^0-9]|$)", r" \1"),  # periods not followed by numbers
        # keep numeric symbols through symbol/diacritic stripping
        lambda text: remove_symbols_and_diacritics(text, keep=".%$¢€£"),
        apply_callable(standardize_numbers),
        apply_callable(standardize_spellings),
        # remove prefix/suffix symbols not adjacent to numbers
        regex_sub(r"[.$¢€£]([^0-9])", r" \1"),
        regex_sub(r"([^0-9])%", r"\1 "),
        regex_sub(r"\s+", " "),  # collapse whitespace
    ]


# --------------------------------------------------------------------------- #
# Per-corpus profiles
# --------------------------------------------------------------------------- #
@dataclass
class NormProfile:
    name: str
    # gold-only: strips this corpus's annotation syntax
    markup_steps: list[Step] = field(default_factory=list)
    # applied to both gold and system, merged on top of SHARED_LEXICON
    lexicon: dict[str, str] = field(default_factory=dict)


def strip_parens_and_slashes() -> list[Step]:
    """Markup shared by both corpora: ``(line notes)`` and ``/unsure/``."""
    return [
        regex_sub(r"\(([^)]+?)\)"),  # remove words between parentheses
        regex_sub(r"/([^)]+?)/"),  # remove words between //
    ]


CORAAL = NormProfile(
    name="coraal",
    markup_steps=[
        regex_sub(r"<[^>]*>"),  # remove non-linguistic noise between <>
        # square brackets denote overlap; drop the brackets, keep the content
        regex_sub(r"[\[\]]"),
        *strip_parens_and_slashes(),
    ],
    lexicon={
        # dialect-specific items with phonetically similar standard variants.
        # NOTE: Koenecke et al. normalize some of these (e.g. "aks").
        r"\baight\b": "alright",
        r"\baks\b": "ask",
        r"\b'bacca\b": "tobacco",
        r"\bbih\b": "bitch",
        r"\bbruh\b": "bro",
        r"\bfella\b": "fellow",
    },
)


SCOSYA = NormProfile(
    name="scosya",
    markup_steps=[
        # remove words between brackets (including {})
        regex_sub(r"[<\[{][^>\]}]*[>\]}]"),
        # strip partial words (single trailing dash); keep whole words (double).
        # NOTE: diminutives (SCOSYA marks with ii) are not handled.
        regex_sub(r"\w+-\s", " "),
        *strip_parens_and_slashes(),
    ],
    lexicon={
        # SCOSYA separates the negative particle from the auxiliary; we convert
        # to "not" to match the "n't" handling in SHARED_LEXICON. The missing
        # particle was transcribed even when not really there, so all negatives
        # can be searched at once.
        r"\bnt\b": "not",
        r"\bna\b": "not",
        # "occasionally 'gaa' has been heard for 'gonna'":
        r"\bgaa\b": "gonna",
        # NOTE: we normalize "nae" -> "not" for now (not fully principled; we
        # do these but not e.g. "yince" -> "once", which is phonetically farther)
        r"\bnae\b": "not",
        r"\bnay\b": "no",
        r"\bnay one\b": "no one",
        r"\bnaybody\b": "nobody",
        r"\bnaywaie\b": "nowhere",
        r"\bnaything\b": "nothing",
    },
)


PROFILES: dict[str, NormProfile] = {
    CORAAL.name: CORAAL,
    SCOSYA.name: SCOSYA,
}


# --------------------------------------------------------------------------- #
# Building a normalizer from a profile
# --------------------------------------------------------------------------- #
def build(profile: NormProfile, *, strip_markup: bool) -> Normalizer:
    """Build a normalizer for ``profile``.

    The only difference between gold and system normalization is markup
    stripping: pass ``strip_markup=True`` for gold transcripts (which carry the
    corpus's annotation syntax) and ``strip_markup=False`` for system output.
    The corpus lexicon is applied either way.
    """
    steps: list[Step] = [str.lower]
    if strip_markup:
        steps += profile.markup_steps
    steps += [
        regex_sub(IGNORE_FILLERS),  # drop filled pauses (gold and system)
        regex_sub(r"\s+'", "'"),  # space before an apostrophe
        # corpus lexicon layered on top of the shared lexicon (shared first);
        # applied to both gold and system to absorb transcription variance.
        replace_lexicon(SHARED_LEXICON | profile.lexicon),
        *numeric_tail(),
    ]
    return Normalizer(steps)


def gold_normalizer(corpus: str) -> Normalizer:
    return build(PROFILES[corpus], strip_markup=True)


def system_normalizer(corpus: str) -> Normalizer:
    return build(PROFILES[corpus], strip_markup=False)
