# Releasing `kazenai`

This procedure is for maintainers publishing the Core package to public PyPI.
Release Core before `kazenai-finops`, because the FinOps 1.1.1 package requires
`kazenai>=1.1.1,<2.0`.

## 1. Prepare one releasable commit

- Confirm `pyproject.toml`, `kazenai/__init__.py`, tests and user-facing docs all
  describe the intended version and supported surface.
- Confirm the working tree contains no secrets, credentials, customer data,
  generated environments or unrelated artifacts.
- Commit the reviewed source and push that commit to the release branch.
- Record the exact commit SHA. Build and tag only that SHA.

Never publish from an uncommitted working tree and never reuse a version that
already exists on PyPI.

## 2. Require CI on all supported Python versions

Wait for the GitHub Actions matrix to pass on Python 3.10, 3.11 and 3.12. The
release gate includes linting, type checks, the full test suite, coverage and
schema/contract verification.

Do not continue if any required job is skipped, cancelled or failing.

## 3. Build from a clean checkout

Create a new checkout of the exact reviewed commit outside the development
working tree:

```bash
git clone https://github.com/kazenai-ai/kazenai-core.git kazenai-core-release
cd kazenai-core-release
git checkout <RELEASE_COMMIT_SHA>
python3 -m venv .venv-release
source .venv-release/bin/activate
python -m pip install --upgrade pip build twine
python -m build
python -m twine check dist/*
```

Confirm that `dist/` was empty before the build and contains only the wheel and
source distribution produced from this commit.

## 4. Test the built artifacts, not the source tree

Install the wheel in another empty virtual environment. Run the package smoke
tests and the supported OpenAI/Anthropic manager smoke tests without adding the
repository to `PYTHONPATH`:

```bash
python3 -m venv .venv-wheel
source .venv-wheel/bin/activate
python -m pip install --upgrade pip
python -m pip install "dist/kazenai-1.1.1-py3-none-any.whl[openai,anthropic]"
python -c "import kazenai; print(kazenai.__version__)"
```

The printed version must be `1.1.1`. Use mocked provider transports or a
non-production tenant for behavioral smoke tests; never place live secrets in
logs.

## 5. Publish Core

Prefer PyPI Trusted Publishing from the protected GitHub release workflow. If a
manual upload is unavoidable, use a narrowly scoped PyPI token from the clean
release environment:

```bash
python -m twine upload dist/*
```

Publishing is irreversible. Check the project name, version, commit SHA and
artifact hashes immediately before approval.

## 6. Verify public PyPI independently

After PyPI serves the release, install it into a fresh environment without
local indexes, editable installs or sibling repositories:

```bash
python3 -m venv .venv-public
source .venv-public/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-cache-dir "kazenai[openai,anthropic]==1.1.1"
python -c "import importlib.metadata as m; print(m.version('kazenai'))"
```

Confirm resolved provider SDK majors remain OpenAI `<2` and Anthropic `<1`.
Rerun the supported package smoke test. Only after this succeeds should
`kazenai-finops==1.1.1` be built and published.

## 7. Tag and document the release

- Tag the exact published commit as `v1.1.1`.
- Create a GitHub Release that links the tested commit and summarizes supported
  paths, compatibility and known limitations.
- Record artifact hashes and the CI run used as release evidence.
- After both public packages pass the clean-install smoke test, update demo pins
  and deploy the matching documentation revision.
