"""Narrow, dependency-free JSON token grammar for the optional Qwen reader.

This is a language constraint, NOT an evidence/semantic correctness checker.
Only the bound ByteLevel BPE tokenizer is supported; no text is repaired.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path

from .model import PLMError

GRAMMAR_VERSION = "plm-qa-byte-dfa-v1"
PREFIX = b'{"answer":"'
ABSTAIN = b',"abstained":true,"citations":[]}'
ANSWER = b',"abstained":false,"citations":['
START = ("prefix", 0)
DONE = ("done",)


def grammar_manifest():
    return {
        "version": GRAMMAR_VERSION,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mechanism": "pre_sampling_byte_dfa_vocabulary_trie_mask",
        "field_order": ["answer", "abstained", "citations"],
        "additional_properties": False, "external_whitespace": False,
        "answer": "empty iff abstained; otherwise nonblank Unicode string",
        "answer_escapes": ["quote", "backslash", "slash", "n", "r", "t"],
        "unicode_escape_sequences": False,
        "citations": "unique IDs selected only from current payload.sources; nonempty iff answering",
        "empty_sources": "schema_forces_abstention_not_learned_semantic_abstention",
        "max_source_ids": 32, "max_source_id_characters": 256,
        "eos": "only after complete object", "postprocessing_repair": False,
        "added_tokens": "all blocked, including tool/thinking markers; EOS only after completion",
        "semantic_correctness": "not_guaranteed",
    }


def source_ids_from_messages(messages):
    """Parse only the actual user payload, never source body text or gold."""
    if len(messages) != 2 or [item["role"] for item in messages] != ["system", "user"]:
        raise PLMError("structured reader requires one system and one user payload")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result
    try:
        payload = json.loads(messages[-1]["content"], object_pairs_hook=unique)
        if not isinstance(payload, dict) or set(payload) != {"question", "question_date", "sources"}:
            raise ValueError("unsupported payload")
        if not isinstance(payload["question"], str) or not isinstance(payload["question_date"], str):
            raise ValueError("unsupported question")
        sources = payload["sources"]
        if not isinstance(sources, list) or len(sources) > 32:
            raise ValueError("unsupported source list")
        result = []
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError("unsupported source")
            value = source.get("source_id")
            if not isinstance(value, str) or not 1 <= len(value) <= 256 or not value.strip():
                raise ValueError("unsupported source ID")
            if any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in value) or value in result:
                raise ValueError("invalid or duplicate source ID")
            result.append(value)
        return tuple(result)
    except (ValueError, TypeError, KeyError):
        raise PLMError("invalid structured reader payload") from None


class JsonAnswerGrammar:
    def __init__(self, source_ids):
        self.ids = tuple(source_ids)
        if len(self.ids) > 32 or len(set(self.ids)) != len(self.ids):
            raise PLMError("invalid grammar citation IDs")
        self.literals = tuple(json.dumps(value, ensure_ascii=True).encode("ascii") for value in self.ids)

    def advance(self, state, data):
        for byte in data:
            state = self.step(state, byte)
            if state is None:
                break
        return state

    def step(self, state, byte):
        phase = state[0]
        if phase in {"prefix", "abstain", "answer_tail"}:
            literal = {"prefix": PREFIX, "abstain": ABSTAIN, "answer_tail": ANSWER}[phase]
            offset = state[1]
            if byte != literal[offset]:
                return None
            if offset + 1 < len(literal):
                return (phase, offset + 1)
            return {"prefix": ("string", False, False, b"", False),
                    "abstain": DONE, "answer_tail": ("citation", b"", ())}[phase]
        if phase == "string":
            _, nonempty, nonblank, pending, escape = state
            if pending:
                lead = pending[0]
                lower, upper = 0x80, 0xBF
                if len(pending) == 1:
                    if lead == 0xE0: lower = 0xA0
                    if lead == 0xED: upper = 0x9F
                    if lead == 0xF0: lower = 0x90
                    if lead == 0xF4: upper = 0x8F
                if not lower <= byte <= upper:
                    return None
                pending += bytes([byte])
                width = 2 if lead < 0xE0 else (3 if lead < 0xF0 else 4)
                if len(pending) == width:
                    return ("string", True, nonblank or not pending.decode("utf-8").isspace(), b"", False)
                return ("string", True, nonblank, pending, False)
            if escape:
                if byte not in b'"\\/nrt':
                    return None
                return ("string", True, nonblank or byte in b'"\\/', b"", False)
            if byte == 34:
                if not nonempty:
                    return ("abstain", 0)
                return ("answer_tail", 0) if nonblank and self.ids else None
            if not self.ids:
                return None
            if byte == 92:
                return ("string", True, nonblank, b"", True)
            if 32 <= byte < 128:
                return ("string", True, nonblank or not chr(byte).isspace(), b"", False)
            if 0xC2 <= byte <= 0xF4:
                return ("string", True, nonblank, bytes([byte]), False)
            return None
        if phase == "citation":
            prefix, used = state[1:]
            prefix += bytes([byte])
            matches = [(idx, literal) for idx, literal in enumerate(self.literals)
                       if idx not in used and literal.startswith(prefix)]
            for idx, literal in matches:
                if prefix == literal:
                    return ("citation_end", used + (idx,))
            return ("citation", prefix, used) if matches else None
        if phase == "citation_end":
            if byte == 93:
                return ("close",)
            if byte == 44 and len(state[1]) < len(self.ids):
                return ("citation", b"", state[1])
            return None
        if phase == "close" and byte == 125:
            return DONE
        return None

    def validate(self, text):
        try:
            return self.advance(START, text.encode("utf-8")) == DONE
        except (UnicodeError, TypeError):
            return False


class ByteVocabulary:
    """Byte strings and a prefix trie; tokenizer construction is once per reader."""
    def __init__(self, token_bytes, eos_ids, vocab_size):
        self.tokens = dict(token_bytes)
        self.eos_ids = frozenset(eos_ids)
        self.vocab_size = vocab_size
        if not self.eos_ids or self.eos_ids & self.tokens.keys() or any(
                type(idx) is not int or not 0 <= idx < vocab_size for idx in self.tokens.keys() | self.eos_ids):
            raise PLMError("invalid constrained vocabulary identity")
        self.root = {}
        for idx, value in self.tokens.items():
            if not isinstance(value, bytes) or not value:
                raise PLMError("empty or invalid constrained token")
            node = self.root
            for byte in value:
                node = node.setdefault(byte, {})
            node.setdefault(-1, []).append(idx)

    @classmethod
    def from_artifacts(cls, path, tokenizer, vocab_size):
        try:
            spec = json.loads((Path(path) / "tokenizer.json").read_text(encoding="utf-8"))
            if spec["model"]["type"] != "BPE" or spec["decoder"]["type"] != "ByteLevel":
                raise ValueError("unsupported tokenizer")
            # GPT-2/Qwen ByteLevel maps all 256 bytes to reversible Unicode labels.
            visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
            labels = visible + list(range(256, 256 + 256 - len(visible)))
            byte_order = visible + [value for value in range(256) if value not in visible]
            decoder = {chr(label): byte for label, byte in zip(labels, byte_order)}
            added = spec.get("added_tokens", [])
            # Qwen marks some tool/thinking added tokens non-special. None may
            # bypass the byte grammar, so mask ALL added IDs, regardless of flag.
            added_ids = {item["id"] for item in added}
            special_ids = {item["id"] for item in added if item.get("special")}
            tokens = {idx: bytes(decoder[char] for char in label)
                      for label, idx in spec["model"]["vocab"].items() if idx not in added_ids}
            eos_ids = set(tokenizer.eos_token_ids)
            if not eos_ids <= special_ids:
                raise ValueError("unbound EOS")
            # Validate actual decoder settings and representative multi-token bytes.
            if getattr(tokenizer, "clean_up_tokenization_spaces", False):
                raise ValueError("tokenizer cleanup unsupported")
            for text in ('{"answer":"中文😀\\\"\\\\",', " line\n\tend", "é𐀀"):
                encoded = tokenizer.encode(text, add_special_tokens=False)
                if b"".join(tokens[idx] for idx in encoded).decode("utf-8") != text:
                    raise ValueError("tokenizer byte mapping mismatch")
                if tokenizer.decode(encoded, clean_up_tokenization_spaces=False) != text:
                    raise ValueError("tokenizer decoder mismatch")
            return cls(tokens, eos_ids, vocab_size)
        except (ValueError, TypeError, KeyError, OSError, AttributeError, UnicodeError):
            raise PLMError("structured output requires the bound Qwen ByteLevel tokenizer") from None


class ConstrainedLogitsProcessor:
    """MLX single-sequence processor. Prompt prefill tail is ignored exactly once."""
    def __init__(self, vocabulary, source_ids, mx, deadline):
        self.vocabulary = vocabulary
        self.grammar = JsonAnswerGrammar(source_ids)
        self.mx = mx
        self.deadline = deadline
        self.state = START
        self.previous = None
        self.cache = OrderedDict()
        self.calls = 0
        self.mask_ms = 0.0

    def allowed_tokens(self, state):
        if state in self.cache:
            self.cache.move_to_end(state)
            return self.cache[state]
        allowed = []
        stack = [(self.vocabulary.root, state)]
        visited = 0
        while stack:
            node, current = stack.pop()
            allowed.extend(node.get(-1, ()))
            for byte, child in node.items():
                if byte == -1:
                    continue
                next_state = self.grammar.step(current, byte)
                if next_state is not None:
                    stack.append((child, next_state))
            visited += 1
            if visited % 4096 == 0 and time.perf_counter() > self.deadline:
                raise PLMError("structured decoding time limit exceeded")
        if state == DONE:
            allowed.extend(self.vocabulary.eos_ids)
        if not allowed:
            raise PLMError("structured grammar has no legal next token")
        self.cache[state] = tuple(allowed)
        if len(self.cache) > 64:
            self.cache.popitem(last=False)
        return self.cache[state]

    def __call__(self, tokens, logits):
        started = time.perf_counter()
        if started > self.deadline:
            raise PLMError("structured decoding time limit exceeded")
        sequence = tuple(tokens.tolist())
        if self.previous is not None:
            if sequence[:-1] != self.previous:
                raise PLMError("unsupported structured decoding sequence lifecycle")
            token = sequence[-1]
            if token in self.vocabulary.eos_ids:
                if self.state != DONE:
                    raise PLMError("premature structured EOS")
            else:
                data = self.vocabulary.tokens.get(token)
                self.state = self.grammar.advance(self.state, data) if data else None
                if self.state is None:
                    raise PLMError("structured generation left the grammar")
        self.previous = sequence
        if tuple(logits.shape) != (1, self.vocabulary.vocab_size):
            raise PLMError("structured logits shape does not match bound vocabulary")
        allowed = self.allowed_tokens(self.state)
        indices = self.mx.array(list(allowed))
        legal_logits = logits[:, indices]
        if not bool(self.mx.any(self.mx.isfinite(legal_logits)).item()):
            raise PLMError("no finite legal structured logits")
        mask = self.mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
        mask[:, indices] = legal_logits
        self.calls += 1
        self.mask_ms += (time.perf_counter() - started) * 1000
        return mask
