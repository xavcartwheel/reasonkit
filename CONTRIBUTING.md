# Contributing

- Install dev deps: `pip install -e ".[test]"`.
- Run tests: `pytest tests/`.
- Keep the package **stdlib-only**. Provider SDKs (openai, anthropic) belong in
  `examples/` only, never in the package or its dependencies.
- Comments are short dev notes, not explanations. One line for the *why*; put
  the *what* in the README or docstrings.
- Bump the version in `pyproject.toml` and `plugins/reasonkit/.claude-plugin/plugin.json`
  together on each release.
