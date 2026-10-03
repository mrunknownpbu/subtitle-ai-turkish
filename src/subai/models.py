from dataclasses import dataclass


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: str | None = None


@dataclass
class Cue:
    start: float
    end: float
    text: str  # lines separated by "\n"
    speaker: str | None = None
