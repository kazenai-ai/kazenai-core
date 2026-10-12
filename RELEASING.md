# Releasing `kazenai`

How maintainers publish `kazenai` to PyPI. Replace `X.Y.Z` with the release
version.

## Order

`kazenai` depends on `kazen-event-schema`, and `kazenai-finops` depends on
`kazenai`. Publish in that order: the schema (when it changed), then
`kazenai`, then `kazenai-finops` at the same version as `kazenai`.

## 1. Prepare the release commit

- Set the version in `pyproject.toml` and `kazenai/__init__.py`, and update the
  install commands in the README and guides.
- Add `RELEASE_NOTES_X.Y.Z.md`.
- Make sure the commit contains no secrets, credentials, customer data, local
  environments or generated files.
- Push the commit to `main`.

Never publish from an uncommitted working tree, and never reuse a version that
already exists on PyPI.

## 2. Wait for CI

CI must pass on Python 3.10, 3.11 and 3.12, including linting, type checks,
the test suite with coverage, the schema-export check and the `frameworks` job.
Do not continue if any job is skipped, cancelled or failing.

Optionally run the paid live smokes before tagging (they never run in CI):

```bash
KAZENAI_LIVE_PROVIDER_SMOKE=1 pytest -m live tests/test_live_provider_smoke.py -q
KAZENAI_LIVE_FRAMEWORK_SMOKE=1 pytest -m live tests/test_live_framework_smoke.py -q
```

The framework smoke also needs a running Agent FinOps and a LiteLLM proxy; see
the test module's docstring.

## 3. Tag and publish

Pushing a `vX.Y.Z` tag runs `.github/workflows/publish.yml`, which builds the
wheel and source distribution from the tagged commit and publishes them with
PyPI Trusted Publishing.

```bash
git tag -a vX.Y.Z -m "kazenai X.Y.Z"
git push origin vX.Y.Z
gh release create vX.Y.Z --title "kazenai X.Y.Z" --notes-file RELEASE_NOTES_X.Y.Z.md --verify-tag
```

## 4. Verify from PyPI

Install the published package into a fresh environment, without local indexes
or editable installs:

```bash
python3 -m venv /tmp/kazenai-release-check
/tmp/kazenai-release-check/bin/pip install --no-cache-dir "kazenai[openai,anthropic]==X.Y.Z"
/tmp/kazenai-release-check/bin/python -c "import kazenai; print(kazenai.__version__)"
```

The printed version must be `X.Y.Z`, and the resolved provider SDKs must stay
within `openai>=1.40,<3` and `anthropic>=0.39,<1`. Then release
`kazenai-finops` at the same version.
