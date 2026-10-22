"""Scientific requests and process-local substitution boundaries."""

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping, Protocol



@dataclass(frozen=True)
class Prompt:
    qid: str
    input_ids: tuple[int, ...]
    category: str = 'unknown'

    def __post_init__(self):
        if not isinstance(self.qid, str) or not self.qid or not self.input_ids:
            raise ValueError('prompt needs a nonempty qid and token sequence')
        if any(type(token) is not int or token < 0 for token in self.input_ids):
            raise ValueError('prompt tokens must be nonnegative integers')


@dataclass(frozen=True)
class DecodeRequest:
    prompt: Prompt
    spec: Mapping[str, Any]
    max_new_tokens: int = 100
    temperature: float = 0.0
    eos_ids: tuple[int, ...] = ()
    stop_on_eos: bool = True
    seed: int = 0
    run_id: str = 'decode'
    sources: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 0:
            raise ValueError('max_new_tokens must be a nonnegative integer')
        if isinstance(self.temperature, bool) or not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError('temperature must be finite and nonnegative')
        if type(self.seed) is not int or not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError('seed/run identity invalid')
        if any(type(token) is not int or token < 0 for token in self.eos_ids):
            raise ValueError('EOS tokens must be nonnegative integers')


@dataclass(frozen=True)
class DecodeResult:
    execution: str
    termination_reason: str
    committed: tuple[int, ...]
    stats: Mapping[str, Any]
    evidence_complete: bool


@dataclass(frozen=True)
class AuditResult:
    source_integrity: bool
    argmax_coverage: bool
    argmax_correctness: bool
    reasons: tuple[str, ...] = ()
    ar_diagnostic: Mapping[str, Any] | None = None


class ModelSession(Protocol):
    device: Any
    dtype: Any
    max_positions: int
    vocab_size: int
    tied_embeddings: bool
    embedding_table: Any

    def embed(self, input_ids: Any) -> Any: ...
    def __call__(self, inputs_embeds: Any, position_ids: Any, **kwargs: Any) -> Any: ...
    def min_mask_value(self) -> float: ...
    def new_cache(self) -> Any: ...
    def cache_len(self, cache: Any) -> int: ...
    def crop_cache(self, cache: Any, keep_len: int) -> None: ...
    def keep_cache(self, cache: Any, prefix_len: int, indices: list[int]) -> None: ...


class MaskPolicy(Protocol):
    needs_hidden: bool
    def init_from_prompt(self, prompt_embeds: Any, generator: Any = None) -> None: ...
    def masks(self) -> Any: ...
    def on_token_committed(self, new_embed: Any, prev_embed: Any) -> None: ...


class ObservationSink(Protocol):
    def write(self, observation: Mapping[str, Any]) -> None: ...


Cancel = Callable[[], bool]

