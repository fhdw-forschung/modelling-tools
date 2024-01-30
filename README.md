# FHDW Modelling Tools

[![Upload Python Package](https://github.com/fhdw-forschung/modelling-tools/actions/workflows/publish.yml/badge.svg)](https://github.com/fhdw-forschung/modelling-tools/actions/workflows/publish.yml)
[![.github/workflows/pytest.yml](https://github.com/fhdw-forschung/modelling-tools/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/fhdw-forschung/modelling-tools/actions/workflows/pytest.yml)
[![.github/workflows/lint.yml](https://github.com/fhdw-forschung/modelling-tools/actions/workflows/lint.yml/badge.svg)](https://github.com/fhdw-forschung/modelling-tools/actions/workflows/lint.yml)

This is a collection of modelling tools. It is intended as a project-independent package.

## Documentation material

Visit the [documentation](https://fhdw-forschung.github.io/modelling-tools/).

There is material for documenation and presentation of research results.
It is is set up using [Jupyter Book](https://jupyterbook.org/en/stable/intro.html).

### Content

The Documentation is sepreated into two parts:

- Directory `docs` ...
- The API reference read from the docstrings

### Generating sites

First of all, it would be best to familiarize with the [official Jupyter Book documentation](https://jupyterbook.org/en/stable/intro.html). It is also helpful to have an understanding of Sphinx, since it is used under the hood.

Manually generate files:

```sh
jb build --all --builder html docs
```

```{note}
When using devcontainers for development (recommended), the generated files are not directly accessible from your systems browser and have to be provided via a web server for example. With `vscode` you could also use the preview functionality of the `"ms-vscode.live-server"` extension (already installed if you use the provided devcontainer configuration of this repo). Alternatively you could utilize Sphinx's `autobuild` extension described below.
```

Auto-generating the docs with a watchdog while developing can be done via the [`sphinx-autobuild` extension (click for details)](https://github.com/executablebooks/sphinx-autobuild). Before running `sphinx-autobuild`, convert the Jupyter Book configuration with the following command (from root of repo):

```sh
jupyter-book config sphinx docs/
```

`> Wrote conf.py to /workspaces/bebefam-exploration/docs`

```{danger}
Do not tinker with the generated configuration (`conf.py`). All configuration is done for Jupyter Book (in `_config.yml`), which then must be converted like shown above, so that configuration is always in sync. For details, [have a look at the repository](https://github.com/executablebooks/sphinx-autobuild).
```

Then run the autobuild with:

```sh
sphinx-autobuild -a --open-browser --watch fhdw/ docs/ docs/_build/html
```

## Contribution

[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/fhdw-forschung/modelling-tools)

- Use the devcontainer.
- Development utilities can be installed via `pipx`
- `poetry` is used for packaging. See [`poetry publish`](https://python-poetry.org/docs/cli/#publish).
