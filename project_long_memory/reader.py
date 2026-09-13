"""Optional fixed local answer reader for evaluation, not the memory runtime.

No automatic downloads, external API, tools, or cross-question chat history.
Only a caller-selected, hash-bound Qwen3 safetensors directory is accepted.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from .model import PLMError


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LocalCausalReader:
    """Fixed MLX/Qwen3 baseline. No model or tokenizer import until prepare()."""

    def __init__(self, model_path: Path, model_id: str, revision: str, seed: int = 0,
                 max_input_tokens: int = 8192, max_generation_seconds: float = 60,
                 output_mode: str = "free"):
        self.path = Path(model_path).expanduser()
        self.model_id = model_id
        self.revision = revision
        self.seed = int(seed)
        self.max_input_tokens = int(max_input_tokens)
        self.max_generation_seconds = float(max_generation_seconds)
        if output_mode not in {"free", "json_schema"}:
            raise PLMError("unsupported local reader output mode")
        self.output_mode = output_mode
        if not (0 <= self.seed < 2 ** 32 and 128 <= self.max_input_tokens <= 24576 and 0 < self.max_generation_seconds <= 300):
            raise PLMError("invalid local reader limits")
        self._manifest = None
        self._model = None
        self._tokenizer = None
        self._signature = None
        self._vocabulary = None

    def _validate_artifacts(self):
        if not self.path.is_dir() or self.path.is_symlink():
            raise PLMError("reader requires an existing local model directory")
        self.path = self.path.resolve()
        for path in self.path.rglob("*"):
            if path.is_symlink() or path.suffix in {".py", ".bin", ".pkl", ".pickle", ".pt"}:
                raise PLMError("reader rejects links, executable model code and pickle weights")
        manifest_path = self.path / "download-manifest.json"
        try:
            downloaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if downloaded["model_id"] != self.model_id or downloaded["revision"] != self.revision:
                raise PLMError("reader model identity does not match download manifest")
            files = downloaded["files"]
            if not isinstance(files, dict) or not {"config.json", "tokenizer.json", "tokenizer_config.json"} <= set(files) or not any(name.endswith(".safetensors") for name in files):
                raise PLMError("incomplete local reader manifest")
            for name, expected in files.items():
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts or not isinstance(expected, str):
                    raise PLMError("unsafe reader manifest path")
                path = self.path / relative
                if not path.is_file() or _hash_file(path) != expected:
                    raise PLMError("reader artifact checksum mismatch")
            config = json.loads((self.path / "config.json").read_text(encoding="utf-8"))
            tokenizer_config = json.loads((self.path / "tokenizer_config.json").read_text(encoding="utf-8"))
            if config.get("model_type") != "qwen3" or config.get("auto_map") or tokenizer_config.get("auto_map"):
                raise PLMError("unsupported or remote-code reader architecture")
            actual = {str(path.relative_to(self.path)) for path in self.path.rglob("*") if path.is_file() and path.name != "download-manifest.json" and not path.name.startswith(".")}
            if not actual <= set(files):
                raise PLMError("reader contains unbound model artifacts")
            return downloaded
        except (OSError, ValueError, KeyError, TypeError):
            raise PLMError("invalid or unavailable reader artifacts") from None

    def _stat_signature(self):
        result = []
        for path in sorted(self.path.rglob("*")):
            if path.is_symlink():
                raise PLMError("reader artifact link appeared after binding")
            if path.is_file():
                stat = path.stat()
                result.append((str(path.relative_to(self.path)), stat.st_size, stat.st_mtime_ns,
                               stat.st_ctime_ns, stat.st_ino))
        return result

    def prepare(self):
        if self._manifest is not None:
            self._assert_unchanged()
            return self
        initial_signature = self._stat_signature()
        artifacts = self._validate_artifacts()
        if initial_signature != self._stat_signature():
            raise PLMError("reader artifacts changed during checksum validation")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        try:
            mx = importlib.import_module("mlx.core")
            runtime = importlib.import_module("mlx_lm")
            mx.set_memory_limit(8 * 1024 ** 3)
            mx.set_cache_limit(512 * 1024 ** 2)
            self._model, self._tokenizer = runtime.load(str(self.path),
                tokenizer_config={"trust_remote_code": False, "local_files_only": True})
            if self.output_mode == "json_schema":
                from .structured_output import ByteVocabulary
                generation = importlib.import_module("mlx_lm.generate")
                if "logits_processors" not in inspect.signature(generation.generate_step).parameters:
                    raise PLMError("MLX runtime does not support structured logits processors")
                config = json.loads((self.path / "config.json").read_text(encoding="utf-8"))
                self._vocabulary = ByteVocabulary.from_artifacts(self.path, self._tokenizer, config["vocab_size"])
        except ImportError:
            raise PLMError("optional MLX reader dependencies are unavailable") from None
        except PLMError:
            raise
        except Exception:
            raise PLMError("local reader initialization failed") from None
        if initial_signature != self._stat_signature():
            raise PLMError("reader artifacts changed during initialization")
        identity = json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode()
        self._manifest = {
            "reader_id": "plm-local-qwen3-mlx-v1", "model_id": self.model_id, "revision": self.revision,
            "artifact_fingerprint": hashlib.sha256(identity).hexdigest(), "backend": "mlx",
            "runtime_versions": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "transformers", "tokenizers")},
            "decoding": {"seed": self.seed, "temperature": 0.0, "enable_thinking": False,
                         "max_input_tokens": self.max_input_tokens, "max_generation_seconds": self.max_generation_seconds},
            "local_files_only": True, "trust_remote_code": False,
            "memory_limit_bytes": 8 * 1024 ** 3, "cache_limit_bytes": 512 * 1024 ** 2,
            "cross_question_prompt_cache": False,
        }
        from .structured_output import grammar_manifest
        self._manifest["decoding"]["output_mode"] = self.output_mode
        self._manifest["structured_output"] = grammar_manifest() if self.output_mode == "json_schema" else {
            "enabled": False, "postprocessing_repair": False}
        self._signature = self._stat_signature()
        return self

    def _assert_unchanged(self):
        if self._signature != self._stat_signature():
            raise PLMError("reader artifacts changed after experiment binding")
        if self._manifest and self._manifest.get("decoding", {}).get("output_mode", self.output_mode) != self.output_mode:
            raise PLMError("reader output mode changed after experiment binding")
        if self.output_mode == "json_schema" and self._manifest:
            from .structured_output import grammar_manifest
            if self._manifest.get("structured_output") != grammar_manifest():
                raise PLMError("reader grammar changed after experiment binding")

    def describe(self) -> Dict[str, Any]:
        self.prepare()
        return json.loads(json.dumps(self._manifest))

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 256) -> Dict[str, Any]:
        self.prepare()
        if not isinstance(messages, list) or not messages or any(
            not isinstance(item, dict) or set(item) != {"role", "content"} or
            item["role"] not in {"system", "user", "assistant"} or not isinstance(item["content"], str)
            for item in messages):
            raise PLMError("reader requires role/content messages only")
        if not 1 <= max_new_tokens <= 2048:
            raise PLMError("invalid reader output limit")
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        import mlx.core as mx
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if len(prompt) > self.max_input_tokens:
            raise PLMError("reader input exceeds bound; no silent evidence truncation")
        mx.random.seed(self.seed)
        started = time.perf_counter()
        constraint = None
        generation_options = {}
        if self.output_mode == "json_schema":
            from .structured_output import ConstrainedLogitsProcessor, source_ids_from_messages
            if self._vocabulary is None:
                raise PLMError("structured vocabulary unavailable; no free-mode fallback")
            constraint = ConstrainedLogitsProcessor(self._vocabulary, source_ids_from_messages(messages),
                                                   mx, started + self.max_generation_seconds)
            generation_options["logits_processors"] = [constraint]
        fragments = []
        final = None
        reason = "stop"
        stream = stream_generate(self._model, self._tokenizer, prompt,
            max_tokens=max_new_tokens, sampler=make_sampler(temp=0), prefill_step_size=256,
            **generation_options)
        try:
            for response in stream:
                final = response
                fragments.append(response.text)
                if time.perf_counter() - started > self.max_generation_seconds:
                    reason = "timeout"
                    break
        finally:
            stream.close()
            mx.clear_cache()
        output_tokens = int(getattr(final, "generation_tokens", 0)) if final is not None else 0
        if reason != "timeout" and output_tokens >= max_new_tokens:
            reason = "length"
        text = "".join(fragments)
        result = {"text": text, "input_tokens": len(prompt), "output_tokens": output_tokens,
                  "latency_ms": (time.perf_counter() - started) * 1000, "finish_reason": reason}
        if constraint is not None:
            valid = constraint.grammar.validate(text)
            from .structured_output import DONE
            if reason == "stop" and (not valid or not constraint.calls or constraint.state != DONE):
                raise PLMError("structured generation did not satisfy the grammar")
            result["structured_output"] = {"mode": self.output_mode, "grammar_valid": valid,
                "constraint_calls": constraint.calls, "mask_ms": constraint.mask_ms,
                "source_ids_sha256": hashlib.sha256(json.dumps(list(constraint.grammar.ids),
                    ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
                "empty_sources_forced_abstention": not bool(constraint.grammar.ids),
                "semantic_correctness": "not_verified", "postprocessing_repair": False}
        self._assert_unchanged()
        return result
