# kazenai 1.1.3

Documentation and packaging release. No change to runtime behaviour or the
public API; upgrading from 1.1.2 is safe.

## Changes

- **Clearer documentation.** The README, quick start, architecture,
  deployment, release and supported-integrations pages are rewritten for
  people using KazenAI; links on the PyPI page now resolve.
- **Smaller source distribution.** The sdist ships the package, licence and
  README only; the test suite lives in the GitHub repository.
- **Cleaner source.** Code comments and docstrings describe behaviour in plain
  language.
- **Standalone tests.** The test suite runs from a checkout of this repository
  alone, with no sibling repositories or private packages.

Requires `kazen-event-schema>=0.6.4,<0.7`. Release together with
`kazenai-finops` 1.1.3.

## Install

```bash
python -m pip install "kazenai[openai]==1.1.3"
# or, for most product integrations:
python -m pip install "kazenai-finops[openai]==1.1.3"
```
