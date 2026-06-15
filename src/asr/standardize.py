"""
Functions to normalize:

    1. Gold transcripts with each other (do we want this?)
    2. Gold transcripts with expected ASR output
    3. All outputs

We use the whisper normalization as a base, and adapt to specifics of the
SCOSYA and CORAAL transcription schemes (e.g., for non-verbal utterances)


NOTE: This might overfit to whisper, so we might want to rerun this tool for
new systems as a sanity check every time we add one.
"""

import csv
from dataclasses import MISSING, fields
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from pandas._libs.tslibs import nattype
import typer
from tqdm import tqdm
from whisper.normalizers.basic import remove_symbols_and_diacritics
from whisper.normalizers.english import (
    EnglishNumberNormalizer,
    EnglishSpellingNormalizer,
)

from asr.chunk import Chunk, TranscribedChunk
from asr.utils import DATA_DIR

app = typer.Typer()


class EnglishTextNormalizer:
    def __init__(self):
        # Added "ah" to ignore_patterns from CORAAL
        self.ignore_patterns = r"\b(hmm|mm|mhm|mmm|uh|um|ah)\b"
        self.replacers = {
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
            # prefect tenses, ideally it should be any past participles, but it's harder..
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
            # reduced forms in CORAAL (might repeat from above)
            r"\bmusta\b": "must have",
            r"\bwoulda\b": "would have",
            r"\bshoulda\b": "should have",
            r"\bcoulda\b": "could have",
            r"\bmighta\b": "might have",
            r"\bgonna\b": "going to",
            r"\bhafta\b": "have to",
            r"\btryna\b": "trying to",
            r"\bsposta\b": "supposed to",
            r"\bfinna\b": "fixing to",  # NOTE: check how this is handled?
            r"\bgotta\b": "got to",
            r"\bwanna\b": "want to",
            r"\boughta\b": "ought to",
            r"\bcause\b": "because",  # NOTE: also not sure how to handle; defaulting to the CORAAL definition
            r"\btil\b": "until",
            r"\b'em\b": "them",
            r"\blemme\b": "let me",
            r"\bi'ma\b": "i am going to",
            r"\bi'm'a\b": "i am going to",
            r"\bwhatchu\b": "what are you",  # could also be "what do you"; how to handle?
            r"\bwhatcha\b": "what are you",
            r"\bgotcha\b": "got you",
            # dialect specific items from CORAAL
            # I selected a subset that have phonetically similar standard
            # variants
            # TODO: decide if we want to keep these; for what it's worth, the
            # Koenecke paper does normalize some of these (e.g., "aks")
            r"\baight\b": "alright",
            r"\baks\b": "ask",
            r"\b'bacca\b": "tobacco",
            r"\bbih\b": "bitch",
            r"\bbruh\b": "bro",
            r"\bfella\b": "fellow",
            # SCOSYA separates the negative particle from the auxiliary verb.
            # we convert to "not" to match the "n't" conversion above.
            # also NOTE: We transcribed the missing negative particle, even
            # though it isn’t really there. This is so that we can search all
            # the negatives at once.
            r"\bnt\b": "not",
            r"\bna\b": "not",
            # "occasionally 'gaa' has been heard for 'gonna'":
            r"\bgaa\b": "gonna",
            # NOTE: we normalize "nae" to "not" for now; do we want to keep?
            # we are not particularly principled about this right now
            # e.g., we do these ones, but not e.g. "yince" -> "once" bc it is
            # phonetically farther
            r"\bnae\b": "not",
            r"\bnay\b": "no",
            r"\bnay one\b": "no one",
            r"\bnaybody\b": "nobody",
            r"\bnaywaie\b": "nowhere",
            r"\bnaything\b": "nothing",
        }
        self.standardize_numbers = EnglishNumberNormalizer()
        self.standardize_spellings = EnglishSpellingNormalizer()

    def __call__(self, s: str, type: str = "default"):
        s = s.lower()

        if type == "coraal":
            # in coraal, square brackets are used to denote overlap.
            # so instead of stripping inside them, we just remove square
            # brackets
            s = re.sub(r"<[^>]*>", "", s)  # remove words between <>
            s = re.sub(r"[\[\]]", "", s)
        elif type == "scosya":
            s = re.sub(
                r"[<\[{][^>\]}]*[>\]}]", "", s
            )  # remove words between brackets (including {})
        else:
            s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets

        if type == "scosya":
            # strip partial words (marked by single trailing dash)
            # but keep whole words (marked by double trailing dashes)
            s = re.sub(r"\w+-\s", " ", s)
            # NOTE: we don't handle diminutives right now (SCOSYA transcribes as
            # marking with ii)

        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = re.sub(r"/([^)]+?)/", "", s)  # remove words between //
        s = re.sub(self.ignore_patterns, "", s)
        s = re.sub(r"\s+'", "'", s)  # when there's a space before an apostrophe

        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        s = re.sub(r"(\d),(\d)", r"\1\2", s)  # remove commas between digits
        s = re.sub(r"\.([^0-9]|$)", r" \1", s)  # remove periods not followed by numbers
        s = remove_symbols_and_diacritics(s, keep=".%$¢€£")  # keep numeric symbols

        s = self.standardize_numbers(s)
        s = self.standardize_spellings(s)

        # now remove prefix/suffix symbols that are not preceded/followed by numbers
        s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        s = re.sub(r"([^0-9])%", r"\1 ", s)

        s = re.sub(r"\s+", " ", s)  # replace any successive whitespaces with a space

        return s


whisper_normalizer = EnglishTextNormalizer()
STD_FUNCTIONS = [("whisper_norm", whisper_normalizer)]


def print_diagnosis(gold_text: list[str], system_text: list[str]):
    exact_match = 0
    total = 0
    for gold, system in zip(gold_text, system_text):
        exact_match += gold == system
        total += 1

    print(f"em: {exact_match} | total: {total}")


@app.command()
def diagnostic(
    corpora: list[str],  # list of corpus names
    subset: int = -1,
):
    print(corpora)
    chunks: list[TranscribedChunk] = []
    same_transcripts = defaultdict(list)
    for corpus in corpora:
        corpus_reader = csv.DictReader(corpus, delimiter="\t")
        corpus_df = pd.read_csv(DATA_DIR / "chunks" / f"{corpus}_chunks.tsv", sep="\t")
        transcript_df = pd.read_csv(
            DATA_DIR / "transcriptions" / f"{corpus}.tsv", sep="\t"
        )
        merged = corpus_df.merge(transcript_df, on="utterance_id", how="left")

        required_fields = set(
            f.name
            for f in fields(TranscribedChunk)
            if f.default is MISSING and f.default_factory is MISSING
        )
        shared_fields = required_fields & set(merged.columns)
        assert len(shared_fields) == len(required_fields), (
            f"Missing fields in merged dataframe: {required_fields - shared_fields}"
        )

        for i, row in enumerate(
            tqdm(merged.to_dict(orient="records"), total=len(merged))
        ):
            transcribed_chunk = TranscribedChunk(**{k: row[k] for k in shared_fields})
            chunks.append(transcribed_chunk)
            same_transcripts[transcribed_chunk.system_text].append(transcribed_chunk)

    gold, system = zip(*((chk.gold_text, chk.system_text) for chk in chunks))
    print_diagnosis(gold, system)

    for name, fn in STD_FUNCTIONS:
        gold = [fn(g, type="scosya") for g in gold]
        system = [fn(s) for s in system]
        print(name)
        print_diagnosis(gold, system)


@app.command()
def standardize():
    pass


if __name__ == "__main__":
    app()
