from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from typing import Any


@dataclass
class RetrievedExample:
    """One nearest-neighbour hit from the vector store.

    `document` is the indexed source text (e.g. the dataset's `whisper_text`),
    `target` is the LaTeX ground truth stored in the metadata. `audio_path` is
    only populated by audio indexes (see generate_audio_rag_dataset.py) and is
    relative to the vector-db root.
    """

    id: str
    document: str
    target: str
    distance: float | None = None
    audio_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_text(self) -> str:
        return f"Original sentence: {self.document}\nLaTeX corrected sentence: {self.target}"


def examples_to_text(examples: list[Any]) -> str:
    """Render retrieved examples as a few-shot block. Accepts RetrievedExample
    objects or pre-formatted strings."""
    rendered = [
        ex.as_text() if isinstance(ex, RetrievedExample) else str(ex) for ex in examples
    ]
    return "\n\n".join(rendered)


_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n?(.*?)\n?```\s*$", re.DOTALL)
_LABEL_RE = re.compile(
    r"^(corrected( sentence| text)?|transcription|latex|output)\s*:\s*", re.IGNORECASE
)


def clean_prediction(text: str) -> str:
    """Strip markdown fences / label prefixes so metrics see only the answer."""
    text = text.strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    text = _LABEL_RE.sub("", text)
    return text.strip()


class BaseGenerator(ABC):
    @abstractmethod
    def generate(
        self, inputs: list[Any], batched_examples: list[list[Any]], **kwargs
    ) -> list[str]:
        """Generate one output string per input.

        `batched_examples[i]` holds the retrieved examples for `inputs[i]`, as
        RetrievedExample objects (or plain strings for backwards compatibility).
        Extra keyword arguments (e.g. `hints`) are optional per-input side
        information; generators that don't use them must accept and ignore them.
        """
