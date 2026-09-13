# External components

The core Python runtime has no third-party runtime dependencies. Optional
components are installed separately; the PLM wheel does not bundle their source,
binaries, model weights or user indexes.

## CodeGraph

- Upstream: https://github.com/colbymchenry/codegraph
- Package: `@colbymchenry/codegraph`
- Integration tested against: `1.5.0`
- Upstream license: MIT
- Copyright notice: Copyright (c) 2026 Colby Mchenry
- Versioned license: https://github.com/colbymchenry/codegraph/blob/v1.5.0/LICENSE
- Used through: an optional local CLI subprocess; existing hosts may instead
  call their own CodeGraph MCP integration.

CodeGraph provides code analysis and graph queries. PLM's adapter provides the
memory/code response assembly and repository-snapshot checks. These are separate
components. The upstream license applies to CodeGraph, not automatically to PLM.

The local acceptance environment used Python 3.9.6, CodeGraph 1.5.0 and the npm
distribution on macOS. Other versions/platforms require their own verification.
See `docs/CODEGRAPH.md` for the supported invocation and validation commands.

## Optional models and evaluation data

The model environments in `requirements-models.txt` and `requirements-reader.txt`
are separate from the zero-dependency runtime. Their dependencies and downloaded
models retain their own licenses. Downloaded evaluation datasets and model weights
are not included in the Python distribution. The code-memory comparison uses
project-authored synthetic fixtures under `examples/code_memory/`.

## Project license status

Apache-2.0 is the proposed PLM license, pending the maintainer's selection. This
inventory does not grant a license to PLM. Add the chosen root `LICENSE` and
matching package metadata before publishing a licensed release.
