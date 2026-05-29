# Publishing to PyPI

The PyPI **distribution name** is [`gpurun-numa`](https://pypi.org/project/gpurun-numa/) because [`gpurun`](https://pypi.org/project/gpurun/) is already used by another GPU job scheduler. The installed console command is still **`gpurun`**.

## Prerequisites

1. [PyPI](https://pypi.org/account/register/) account
2. [TestPyPI](https://test.pypi.org/account/register/) account (recommended for first upload)
3. API token: PyPI → Account settings → API tokens → “Add API token” (scope to project `gpurun-numa` when it exists)

## One-time setup

```bash
pip install "gpurun-numa[dev]"
# or: pip install build twine pytest
```

Store credentials locally (optional):

```bash
# ~/.pypirc — do not commit this file
[pypi]
username = __token__
password = pypi-AgENdGVzdC5weXBpLm9yZwI0...

[testpypi]
username = __token__
password = pypi-AgENdGVzdC5weXBpLm9yZwI0...
```

Prefer environment variables in CI:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...
```

## Release checklist

1. Update version in `pyproject.toml` and `src/gpurun/__init__.py`
2. Update `CHANGELOG.md`
3. Run tests: `pytest`
4. Build:

   ```bash
   rm -rf dist/ build/ src/*.egg-info
   python -m build
   ```

5. Check the wheel:

   ```bash
   pip install dist/gpurun_numa-*.whl
   gpurun --help
   ```

6. Upload to **TestPyPI** first:

   ```bash
   python -m twine upload --repository testpypi dist/*
   pip install -i https://test.pypi.org/simple/ gpurun-numa
   ```

7. Upload to **PyPI**:

   ```bash
   python -m twine upload dist/*
   ```

8. Tag the release:

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

PyPI does not allow re-uploading the same version string.

## Trusted publishing (GitHub Actions)

The workflow in `.github/workflows/publish.yml` can publish on GitHub Release without storing tokens in secrets.

1. Push the repo to GitHub
2. On PyPI: project **gpurun-numa** → Publishing → Add a new trusted publisher
   - Owner: your GitHub user/org
   - Repository: `gpurun-numa` (or your repo name)
   - Workflow: `publish.yml`
   - Environment: (leave empty) or `pypi` if you use a GitHub environment

3. Create a GitHub Release from tag `v0.1.0` — the workflow uploads automatically

Manual dispatch is also available from the Actions tab.

Optional: add a GitHub **environment** named `pypi` in the workflow for approval gates before publish.

### Manual upload (no trusted publishing)

```bash
pip install build twine
python -m build
TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-... python -m twine upload dist/*
```

## Update repository URLs

Repository: https://github.com/spacejake/gpurun-numa
