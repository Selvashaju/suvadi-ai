from dataclasses import dataclass, asdict


@dataclass
class OCRToken:
    """
    One word detected by Tamil OCR.
    """

    text: str
    confidence: float
    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict:
        return asdict(self)