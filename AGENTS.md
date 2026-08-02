# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

## Core Principles (CRITICAL)

**Less is more. The simplest solution is the best solution.** The action hierarchy for every change: **Delete > Replace > Add**.

1. **Solve at the owner**: Put behavior in the code path that owns or observes it. For fixes, never guard a symptom with a staleness check, initialization flag, skip-first-call branch, or `try/except` around broken logic; relocate the trigger and delete the wrong path. For features, extend the existing owner rather than creating a parallel abstraction.
2. **Search and reuse first**: Search the whole repository before creating a feature, component, helper, workflow, or utility. Reuse or adapt what exists, consolidate in-scope duplication in the shared owner, and delete duplicate paths. Three similar lines beat a helper nobody else calls.
3. **Delete and modify existing code before creating new code**: Bugfixes are net-negative by default unless deletion and relocation are demonstrably impossible. A new file must first prove it cannot fit cleanly in an existing owner.
4. **Keep scope minimal**: Implement only the simplest complete solution. Avoid impossible-state handling, speculative flags, compatibility shims, policy scaffolding, and unrelated cleanup. Tests are out of scope by default — rely on existing coverage and focused validation; only an uncovered, high-risk regression path justifies minimal new test code.
5. **Ship zero-regression, production-ready changes**: Understand what you remove instead of retaining broken code as insurance. Remove unused imports, functions, types, files, and comments; run relevant cleanup checks; and thoroughly debug and validate the changed owner. Do not break existing features or workflows unless the PR intentionally removes them with evidence.

**Review gate:** for every addition, the reviewer decides whether deleting or changing existing code would have fixed the problem instead — if it would, that is a blocking finding. A missing or thin PR description is never itself a finding.

NEVER push to `main`. NEVER force push. Always start work in a new git worktree (`git worktree add`) on a feature branch and open a PR — never edit the primary checkout directly, it may hold in-flight work.

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Review the full diff in-session against the Core Principles, performance, and the review gate above, then batch the fixes into one commit and push. After each round of bot or human commits, pull and resume the same reviewer on `<last-reviewed-sha>..HEAD` plus anything that delta could have invalidated. Repeat until the local head matches the live head.
3. Hand off or merge only on a clean final pass: one cold full-diff review returning LGTM with no findings, on a head that is still live at merge time.
4. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never reset or revert commits you did not author.
5. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

## Commands

```bash
# Install dependencies (use uv, never bare `pip install`); CI also adds `pytest ultralytics` and `--system`
uv pip install -r requirements.txt pytest

# Byte-compile every file — CI's first gate, catches syntax errors before tests (CI runs `uv run python -m compileall -q .`)
python -m compileall -q .

# Run all tests (CI runs `uv run pytest -q`)
pytest -q

# Single file / single test
pytest tests/test_converters.py
pytest tests/test_converters.py::test_coco_conversion_writes_keypoints -v

# Format and lint (no in-repo config; Ultralytics Actions applies its own Ruff + docformatter settings)
ruff format . && ruff check --fix .
```

- CI (`ci.yml`) is a single job on `ubuntu-latest` / Python 3.11 (plus a daily 08:00 UTC schedule); it byte-compiles with `compileall`, then runs `pytest -q`. There is no test matrix; the README states Python 3.8+ but CI exercises only 3.11.
- There is no `pyproject.toml`, `setup.py`, or in-repo Ruff config — this is a script collection, not an installable package, so `ruff`/`docformatter` run with Ultralytics Actions' own settings and the bot's output can differ from a bare local `ruff` run.
- `tests/test_converters.py` calls the `convert_*` functions directly on synthetic `tmp_path` fixtures and never touches the network, so the suite runs fully offline.

## Architecture

JSON2YOLO is a small set of standalone scripts that convert third-party annotation JSON into Ultralytics YOLO label files under `--save-dir` (a `labels/` tree plus optional copied `images/`). COCO and LabelMe also write a dataset YAML; the legacy `infolks`/`vott`/`ath` converters instead emit Darknet-style `.names` and train/test split `.txt` files (INFOLKS and ATH also write a `.data` file, VoTT does not). It is superseded by `convert_coco()` in the main `ultralytics` package — the README banner points users there; this repo is maintained but no longer actively extended.

- `general_json2yolo.py` — the entry point: `parse_args()` exposes `--source {COCO,LabelMe,infolks,vott,ath}` dispatched in `__main__`, and each source has a `convert_*_json` function. COCO and LabelMe are the actively-used paths (segments, keypoints, COCO RLE via `rle2polygon`, LabelMe base64 masks via `mask2points`, multi-part segment stitching via `merge_multi_segment`); `infolks`, `vott`, and `ath` are legacy converters kept for older workflows.
- `labelbox_json2yolo.py` — a separate Labelbox converter (`convert`, `load_labelbox_json`); its `__main__` calls `convert()` with a hardcoded export filename rather than the CLI, and `load_labelbox_json` accepts a JSON list, a single JSON object, or newline-delimited JSON.
- `utils.py` — the only shared module, imported by both scripts: `make_dirs` deletes and recreates the output dir on every run (so `--save-dir` is destructive), plus `exif_size`, the train/test split helpers, and `coco91_to_coco80_class`.

## Conventions

- Every Python file starts with `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` — Ultralytics Actions adds headers automatically; don't add or revert them manually.
- Google-style docstrings; the Actions bot runs Ruff, docformatter, prettier (YAML/JSON/Markdown), and codespell on PRs and its formatting can differ from local — expect bot commits on the branch, and `git pull --rebase` before pushing more.
- Tests are offline and deterministic: build synthetic COCO/LabelMe/VoTT JSON with `tmp_path` and assert exact label strings; Labelbox is covered only at the `load_labelbox_json` NDJSON-parse level, not full conversion — do not add tests that download weights or hit the network.
- No package version and no release process — this is a script repo with no `__version__` to bump; there is nothing to publish.
