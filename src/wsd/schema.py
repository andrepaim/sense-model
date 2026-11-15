"""Portable dictionary-selection records. Offsets are Unicode code points."""

from dataclasses import asdict, dataclass, field


@dataclass
class Candidate:
    id: str
    definition: str


@dataclass
class Target:
    start: int
    end: int
    text: str


@dataclass
class Example:
    id: str
    text: str
    target: Target
    candidates: list[Candidate]
    language: str = "en"
    definition_language: str = "en"
    gold_ids: list[str] | None = None
    provenance: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("example must be an object")
        value = dict(value)
        value["target"] = Target(**value["target"])
        value["candidates"] = [Candidate(**c) for c in value["candidates"]]
        example = cls(**value)
        example.validate()
        return example

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("example id is required")
        if not isinstance(self.text, str) or not isinstance(self.target, Target):
            raise ValueError("text must be a string and target a Target")
        if not isinstance(self.candidates, list) or any(not isinstance(c, Candidate) for c in self.candidates):
            raise ValueError("candidates must be a list of Candidate objects")
        t = self.target
        if type(t.start) is not int or type(t.end) is not int or not 0 <= t.start < t.end <= len(self.text):
            raise ValueError("invalid target offsets")
        if self.text[t.start : t.end] != t.text:
            raise ValueError("target text does not match offsets")
        ids = [c.id for c in self.candidates]
        if len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i for i in ids):
            raise ValueError("candidate IDs must be unique nonempty strings")
        if any(not isinstance(c.definition, str) or not c.definition.strip() for c in self.candidates):
            raise ValueError("candidate definitions cannot be empty")
        if len(ids) > 127:
            raise ValueError("at most 127 candidates supported")
        if self.gold_ids is not None and (
            not isinstance(self.gold_ids, list) or any(i not in ids for i in self.gold_ids)
        ):
            raise ValueError("gold IDs must refer to supplied candidates")
        if (
            not isinstance(self.language, str)
            or not self.language
            or not isinstance(self.definition_language, str)
            or not self.definition_language
            or not isinstance(self.provenance, dict)
        ):
            raise ValueError("language and provenance are required")


def utf16_to_codepoint(text: str, offset: int) -> int:
    """Convert a JS UTF-16 offset, refusing a split surrogate pair."""
    if type(offset) is not int or offset < 0:
        raise ValueError("invalid UTF-16 offset")
    raw = text.encode("utf-16-le")
    if offset * 2 > len(raw):
        raise ValueError("offset out of bounds")
    try:
        return len(raw[: offset * 2].decode("utf-16-le"))
    except UnicodeDecodeError as exc:
        raise ValueError("offset splits surrogate pair") from exc
