## Git Techniques
Git is far more powerful than the routine add–commit–push loop. Below are two practical workflows that help keep your history tidy and track down regressions quickly.

### Merge a Divergent Feature Branch
Sometimes a long-running `dev` branch carries dozens of experimental commits, but you want a single polished commit on `main`. A careful squash-and-cherry-pick flow keeps history readable and limits risk on `main`.

1. **Inspect the history.** Find the first commit where real work begins so you know what to keep. A quick view is `git log --oneline main..dev`.
2. **Squash the work.** Soft-reset to the commit just before the real work, which leaves your changes staged, then create the clean commit.

   ```bash
   git checkout dev
   git reset --soft <commit_before_changes>
   git commit -m "Describe the feature or fix"
   ```

3. **Prepare a merge branch.** Start a fresh branch from `main` so you integrate on top of the latest production-ready code.

   ```bash
   git checkout main
   git checkout -b change_to_merge
   git cherry-pick <new_squashed_commit>
   ```

4. **Resolve and review.** Fix any conflicts, run your test suite, and ensure the branch builds cleanly before opening a pull request or merging.

This pattern gives reviewers a focused diff while preserving the exploratory history on `dev` in case you need it later.

### Find the Commit That Introduced a Bug
When a regression appears, `git bisect` halves your search space until the culprit commit surfaces. You only need to tell Git which commit was good and which was bad; the tool guides you through the rest.

```bash
git bisect start
git bisect bad <bad_commit_hash>    # commit where the bug reproduces
git bisect good <good_commit_hash>  # oldest commit known to be bug-free
```

Git now checks out a midpoint between the two. Test the project—run the scenario manually or script it. Mark the result so Git can narrow the range:

```bash
git bisect bad     # bug still present
git bisect good    # bug not present
```

Repeat the test after each checkout. When Git reports the first bad commit, note its message, author, and context so you can craft a fix or revert. When finished, return to your previous branch:

```bash
git bisect reset
```

Tip: if you can script the test (for example `./run-tests.sh`), use `git bisect run ./run-tests.sh` to automate the entire process.
