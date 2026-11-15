# Pilot implementation verification

This receipt describes software tests, not completed annotation or model quality.
No real user API key or paid Jev request was used.

- SQLite tests cover immutable initial votes, source-gold stripping, conflicts,
  blind second reviewers, AI gating/exposure, two-person adjudication, exact
  JSONL checksums, source provenance, demo exclusion and usage accounting.
- HTTP tests cover Host/Origin/CSRF checks, session-only credentials, no key in
  SQLite or backups, key clearing, assistance caching, blinded downloads and
  suppression of aggregate label statistics before independent review.
- Jev adapter tests mock official System One responses, authentication/rate-limit
  errors, invalid distributions, unknown candidates, redirect rejection and
  source/vote/key exclusion from metadata. Actual credentials remain unverified.
- Collection tests cover repeated words, Unicode offsets, phrase overlaps, full
  candidate inventories, declared source/definition languages, deterministic
  sampling, caps, manifests and honest shortfalls.
- Playwright runs the real local UI/server/SQLite and mocks only the paid Jev
  adapter. It exercises import, unsafe source HTML rendered inert, exact Unicode
  highlighting, saved independent vote, key entry, provider failure without vote
  loss, suggestion display, post-AI revision, blind reviewer switch, disagreement
  adjudication, gold/manifest downloads with matching checksum, key clearing and
  mobile overflow checks. Browser-test records are temporary authored fixtures.
- The Python wheel was built and verified to contain all three static UI assets.

Commands:

```sh
uv sync --frozen --extra train --extra export --extra dev
uv run ruff check src tests scripts web-tests/server_fixture.py
uv run pytest -q
cd web-tests
npm ci
npx playwright install chromium
npm test
```

CI includes the existing model suite and a separate annotation browser job with
Chromium installed on the runner. Local SQLite workspaces, keys and test outputs
are ignored by Git. Reviewer names are local workflow identities, not accounts.
The filesystem owner can inspect the workspace; blindness prevents accidental
in-app disclosure, not an adversary with local disk access.
