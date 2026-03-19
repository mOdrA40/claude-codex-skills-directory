# Go Tooling (Senior Defaults)

## Baseline Commands (No Extra Tooling)

- Format (recommended): `gofmt -w $(go list -f '{{.Dir}}' ./...)` (PowerShell: `gofmt -w @(go list -f '{{.Dir}}' ./...)`)
- Vet: `go vet ./...`
- Test: `go test ./...` (+ `-race` when concurrency matters)

## `golangci-lint` (v2+ safe config notes)

When using `golangci-lint` v2:

- Prefer `golangci-lint run --fast-only` during local iteration; run full `golangci-lint run` in CI.
- If you previously used `linters.fast`, it’s removed in v2; replace with `linters.default: fast` or use `--fast-only`.
- If you previously used `run.skip-files`, migrate to:
  - `linters.exclusions.paths` (exclude lint issues by file path)

### Excluding false positives (preferred patterns)

```yaml
linters:
  exclusions:
    rules:
      - path: '(.+)_test\\.go'
        linters:
          - funlen
          - goconst
```

### Suggested baseline linters (keep it boring)

Start with a small set and expand only when it buys real safety:

- Correctness: `govet`, `staticcheck`, `errcheck`, `ineffassign`, `unused`
- Errors: `errorlint`
- Context/timeouts (services): `noctx`
- Security (services): `gosec` (tune to reduce noise)

Rule of thumb: if a linter is noisy, either tune it or remove it—don’t teach the team to ignore warnings.

## Vulnerability Scanning

If you ship binaries or services, add Go’s vulnerability scan to release gates:

- `govulncheck ./...`
- `govulncheck -test ./...` (includes test dependencies; can be noisy but useful)

## CI Recommendations (Minimal)

- Always: `go test ./...`
- Services/concurrency: `go test -race ./...` (where supported)
- Lint: `golangci-lint run` (cache enabled in CI if possible)
