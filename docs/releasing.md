# Releasing vectormark to PyPI

vectormark publishes via **PyPI Trusted Publishing** (OIDC) — GitHub Actions
proves the workflow's identity to PyPI directly, so there is **no API token** in
this repository. The release workflow lives at `.github/workflows/release.yml`
and runs on any `v*` tag.

## One-time setup (maintainer)

You only do this once, on pypi.org. It cannot be automated from here — it ties
your PyPI account to this repo.

1. Create the project's trusted publisher **before the first upload** using a
   *pending publisher* (PyPI lets you pre-register a project that does not exist
   yet):
   - Log in to <https://pypi.org> → **Account → Publishing → Add a pending publisher**.
   - **PyPI Project Name:** `vectormark`
   - **Owner:** `pradeepmouli`
   - **Repository name:** `vectormark`
   - **Workflow name:** `release.yml`
   - **Environment name:** `pypi`
2. In this GitHub repo: **Settings → Environments → New environment** → name it
   `pypi`. (Optional: add a required reviewer so a human approves each publish.)

That's it. The `id-token: write` permission and the `environment: pypi` binding
in the workflow match what you registered above.

## Cutting a release

1. Bump `version` in `pyproject.toml` (PyPI versions are immutable — a number can
   never be reused, so never re-tag an already-published version).
2. Commit on `master`.
3. Tag and push:
   ```bash
   git tag v0.0.1
   git push origin v0.0.1
   ```
4. The `Release` workflow builds the sdist + wheel, runs `twine check`, and
   publishes to PyPI. Watch it under the repo's **Actions** tab.

## Local build check (optional, before tagging)

```bash
uv build                 # -> dist/vectormark-<ver>-py3-none-any.whl + .tar.gz
uvx twine check dist/*   # validate metadata renders on PyPI
```

vectormark's own wheel is pure Python (`py3-none-any`); the heavy dependencies
(numpy/scipy/scikit-image/shapely/pillow) and the optional `scoring` extra
(resvg-py) all ship prebuilt manylinux/musllinux/macos/win wheels, so
`pip install vectormark` works in any networked environment.

> **Note:** the MCP App widget HTML (`integrations/mcp-app/dist/mcp-app.html`)
> lives outside the Python package, so a pip-installed `vectormark[server]` falls
> back to a build-stub widget. The server and CLI work regardless; only the
> bundled React preview is absent. If the hosted server is ever shipped via
> PyPI, move the built HTML into package data (`src/vectormark/_widget/`) and
> `force-include` it.
