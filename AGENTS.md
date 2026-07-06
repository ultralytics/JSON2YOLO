# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

## Core Principles (CRITICAL)

**Delete > Replace > Add.** Before writing any change, answer in order: what can I delete? what can I replace? only then, what must I add?

The most common agent failure in this repo is reaching for the locally-safest edit — a new guard, flag, or helper — instead of fixing ownership. These tripwires override that instinct:

1. **Never guard a symptom — relocate the trigger.** A fix that adds a condition to suppress bad behavior (a staleness check, an is-initialized flag, a skip-first-call guard, a try/except around broken logic) is wrong by default. Find the code path that should own the behavior, move the logic there, and delete the code that got it wrong. Example: a warning fired from stale state; the right fix was not a recency guard — it deleted the stale detection and moved the trigger into the code path that observes the event live.
2. **Bugfixes are net-negative by default.** A bugfix that adds more lines than it removes needs a one-sentence justification in the PR body naming why deletion and relocation were impossible.
3. **Search the repo before creating anything.** Before building a feature or helper, search the whole repo — it likely exists (`utils.py` holds the shared helpers imported by both converters). If two modules grow the same logic, consolidate into `utils.py` and delete the duplicates. Avoid premature abstraction — three similar lines beat a helper nobody else calls.
4. **Deletion beats caution.** Zero regression means understanding the code you remove, not leaving it in place as insurance. Keeping broken or duplicated code "to be safe" is itself the regression: it is how repos rot. All changes must still ship debugged, validated, and production ready.

**Output gate:** every PR body must contain a `Deleted:` line naming the code removed (functions, branches, files, config). Features must name what they reused or consolidated. `Deleted: nothing` demands the rule-2 justification.

**Review gate:** adversarial reviewers must answer two questions before LGTM: (a) what could have been deleted instead of added? (b) does any added condition suppress a symptom rather than relocate a trigger? A finding on either blocks LGTM.

**This file is code — additions require deletions.** To add a rule here, remove or merge one. When everything is emphasized, nothing is.

**NEVER push to `main`. NEVER force push.** Always start work in a new git worktree (`git worktree add`) on a feature branch and open a PR — never edit the primary checkout directly, it may hold in-flight work.

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Launch an independent adversarial review agent with cold context (just the PR diff and this file) to hunt for bugs, regressions, and Core Principles violations — use the Codex CLI, one fresh `codex exec` run per round. Fix, push, and repeat until a fresh run reports LGTM.
3. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never force-push, reset, or revert commits you did not author.
4. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

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
