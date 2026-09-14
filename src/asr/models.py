"""
Transcription models under evaluation.

A model is anything with a ``name`` and a ``transcribe(audio_path) -> str``
method. Implement that protocol and add one ``MODELS`` entry to make a new
system available to the transcription/evaluation pipeline.
"""

from datasets import Audio, Dataset
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Model(Protocol):
    name: str

    def transcribe(self, audio_path: str | Path) -> str:
        """Transcribe a single audio clip and return its text."""
        ...


class WhisperModel:
    def __init__(
        self, size: str = "large", language: str | None = None, device: str = "cuda:0"
    ):
        self.size = size
        self.device = device
        self.name = f"whisper-{size}-{language}"
        self.language = language
        self._model = None  # loaded lazily so importing is cheap

    def _ensure_loaded(self):
        if self._model is None:
            import whisper

            self._model = whisper.load_model(self.size, device=self.device)
        return self._model

    def transcribe(self, audio_path: str | Path) -> str:
        model = self._ensure_loaded()
        result = model.transcribe(str(audio_path), language=self.language)
        return str(result["text"])


class HFModel:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        from transformers import pipeline

        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model=model_path,
            device=device,
        )
        self.name = model_path

    def transcribe(self, audio_path: str | Path) -> str:
        try:
            output = self.transcriber(str(audio_path))
            return str(output["text"])  # type: ignore
        except:
            return ""


# Registry of models available for evaluation. Each entry is a factory taking a
# ``device`` so the data-parallel transcriber can place a copy on each GPU.
MODELS: dict[str, Callable[..., "Model"]] = {
    "whisper-large": lambda device="cuda:0": WhisperModel("large", device=device),
    "whisper-large-en": lambda device="cuda:0": WhisperModel(
        "large", language="en", device=device
    ),
    "wav2vec": lambda device="cuda:0": HFModel(
        "facebook/wav2vec2-large-960h", device=device
    ),
    "ibm-granite": lambda device="cuda:0": HFModel(
        "ibm-granite/granite-speech-4.1-2b", device=device
    ),
    "nvidia-parakeet": lambda device="cuda:0": HFModel(
        "nvidia/parakeet-tdt-0.6b-v3", device=device
    ),
    "nvidia-fastconformer": lambda device="cuda:0": HFModel(
        "nvidia/stt_en_fastconformer_ctc_large", device=device
    ),
    "cohere-transcribe": lambda device="cuda:0": HFModel(
        "CohereLabs/cohere-transcribe-03-2026", device=device
    ),
}


def build_model(name: str, device: str = "cuda:0") -> "Model":
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}; available: {list(MODELS)}")
    return MODELS[name](device=device)
