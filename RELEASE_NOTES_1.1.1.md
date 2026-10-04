# kazenai 1.1.1

Compatibility/distribution patch on the Control 1.1.x line. **No runtime capability
matrix widening.**

## What changed and why

- Added optional provider extras so first-run installs cannot silently resolve
  uncertified OpenAI 3.x or Anthropic 1.x majors.
- Synchronized active install guidance, release docs, and package metadata to 1.1.1.
- Marked the thin `kazenai-finops` re-export cell as `fixture-verified` (identity only).

## Install

```bash
python -m pip install "kazenai[openai]==1.1.1"
python -m pip install "kazenai[anthropic]==1.1.1"
# Most product integrations should use:
python -m pip install "kazenai-finops[openai]==1.1.1"
```

## Supported provider methods and SDK ranges

- Sync OpenAI Chat Completions: non-streaming, `stream=True`, `.stream()` helper
- Sync Anthropic Messages: non-streaming, `stream=True`, `.stream()` helper
- Certified ranges: `openai>=1.40,<2`, `anthropic>=0.39,<1`
- Python 3.10–3.12

## Explicitly unsupported

- Async clients
- OpenAI Responses API
- OpenAI Realtime / WebSocket
- Bedrock / Vertex wrappers
- Later provider SDK majors outside the certified ranges

## Behavior change

None intended versus 1.1.0 certified surfaces.

## Security / privacy impact

None. Capture remains metadata-default; bodies require explicit opt-in with consent.

## Upgrade from 1.1.0

```bash
python -m pip install -U "kazenai[openai]==1.1.1"
```

Prefer extras over bare `pip install openai` / `anthropic`.

## Links

- Docs: https://docs.kazenai.com/
- Issues: https://github.com/kazenai-ai/kazenai-core/issues
- Source tag: `v1.1.1` (fill commit SHA at publication)

## Known limitation

Streaming cancellation / missing authoritative usage remains financially
**pending/unknown** rather than exact zero. Client close does not prove the
provider stopped billing.

## Artifact digests

Record wheel/sdist SHA-256 and PyPI provenance links after Trusted Publishing.
