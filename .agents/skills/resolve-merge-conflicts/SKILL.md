---
name: resolve-merge-conflicts
description: "Use when you need to resolve an in-progress git merge/rebase conflict."
---

Adapted from [mattpocock/skills](https://github.com/mattpocock/skills) (`resolving-merge-conflicts`).

1. **See the current state** of the merge/rebase. Check git history, and the conflicting files.

2. **Find the primary sources** for each conflict. Understand deeply why each change was made, and what the original intent was. Read the commit messages, check the PRs, check original issues at github.com/pik-piam/industry_issues. Also search the tree for changes that conflict in meaning but not in text (e.g. a renamed function, config key or dimension still used under the old name by the other side).

3. **Resolve each hunk.** Preserve both intents where possible. Do **not** invent new behaviour. Never `--abort`.

4. **Ask when ambiguous.** If the intents are incompatible, unclear, or the resolution would change model results (`config/`, `config/scenarios/`, model logic), do not guess: ask the user with `AskUserQuestion`. State what each side changed and why, offer concrete resolutions and a recommendation. Collect the ambiguous hunks and ask about them together, and resolve the unambiguous ones in the meantime.

5. **Repo-specific files.**
   - `uv.lock`: take either side, resolve `pyproject.toml`, then run `uv lock`.
   - Generated docs (`docs/<model>/config_schema.md`, `docs/<model>/definitions/*`, `docs/<model>/assumptions.md`): never edit by hand. Take either side and tell the user to regenerate them via a docs export of a model run.
   - `data_in/legacy` submodule: ask the user which commit to use.

6. **Run the checks:** `uv check`, then `uv run pytest`. Fix anything the merge broke. Do not run formatters or pre-commit.

7. **Finish the merge/rebase.** Stage everything and commit. If rebasing, continue the rebase process until all commits are rebased. Do not push. Report the decisions made and any follow-ups (e.g. regenerating docs).
