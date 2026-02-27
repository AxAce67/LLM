# Migration Branch Strategy

Use two long-lived branches:

- `main`: stable production branch
- `migration-ollama`: migration work branch for HF/Ollama compatibility

Recommended flow:

1. Create migration branch from latest stable `main`.
2. Implement and test migration changes only in `migration-ollama`.
3. Keep `main` receiving only hotfixes and operational patches.
4. Rebase migration branch regularly on top of `main`.
5. Merge into `main` only after:
   - e2e pipeline test passes
   - model export validation passes
   - rollback procedure is verified

Rollback policy:

- Keep previous app package and previous model package.
- If migration release fails, restore using:
  - `ops/restore_local.sh`
  - previous `version.json` and package artifacts.
