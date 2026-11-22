from typing import TypedDict, Union

GateInput = Union[str, bytes]

class OptionScore(TypedDict):
    label: str
    score: float

class Decision(TypedDict):
    choice: str
    confidence: float
    scores: list[OptionScore]
    cached: bool
    latency_ms: int

class YesNo(TypedDict):
    answer: bool
    probability: float
    confidence: float
    cached: bool
    latency_ms: int

class Score(TypedDict):
    level: str
    expected: float
    confidence: float
    legend: list[str]
    scores: list[OptionScore]
    cached: bool
    latency_ms: int

class Gate:
    def __init__(
        self,
        cache_dir: str | None = None,
        cache_entries: int = 4096,
        threads: int = 2,
    ) -> None: ...
    def cache_path(self) -> str: ...
    def decide(self, input: GateInput, choices: list[str]) -> Decision: ...
    def yes_no(self, input: GateInput, question: str) -> YesNo: ...
    def score(self, input: GateInput, levels: list[str]) -> Score: ...
    def decide_batch(self, inputs: list[GateInput], choices: list[str]) -> list[Decision]: ...
    def yes_no_batch(self, inputs: list[GateInput], question: str) -> list[YesNo]: ...
    def score_batch(self, inputs: list[GateInput], levels: list[str]) -> list[Score]: ...
