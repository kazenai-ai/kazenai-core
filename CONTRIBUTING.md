# Contributing

## License
By contributing, you agree that your contributions are licensed under the Apache License, Version 2.0, unless a separate agreement states otherwise.

## Development
Use Python 3.10+; CI verifies Python 3.10, 3.11 and 3.12, and a separate `frameworks` job runs the LangChain/LangGraph adapter tests on the pinned framework versions (`pip install -e ".[dev,langchain,langgraph]"` to run them locally). Prefer tests that do not require private sibling checkouts. Do not commit secrets, `.env` files, or private customer data.

Paid live smokes (`tests/test_live_provider_smoke.py`, `tests/test_live_framework_smoke.py`) are opt-in, marked `live`, and never run in CI. Keep provider keys outside the repository.

Maintainers must follow [RELEASING.md](RELEASING.md) and publish only artifacts
built from the exact clean commit that passed the full CI matrix.

## Code of conduct
Be respectful. No harassment or spam.
