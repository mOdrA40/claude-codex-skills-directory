# Errors (Taxonomy, Wrapping, and API Mapping)

## Goals

- Callers can reliably branch using `errors.Is/As`.
- Logs contain context once (no double-logging).
- Public APIs expose stable `code` + appropriate HTTP status (see `http-api.md`).

## Golden rules

- Add context at boundaries: `fmt.Errorf("doing X: %w", err)`
- Never wrap with `%v` when you mean `%w` (you lose `errors.Is/As`).
- Don’t log-and-return the same error at the same layer.
- Don’t use `panic` for expected failures (use it for truly impossible invariants).

## Sentinel vs typed errors

### Sentinel (simple branch)

```go
var ErrNotFound = errors.New("not found")

if errors.Is(err, ErrNotFound) { ... }
```

Use when:
- you only need “this class of error happened”
- no extra structured data needed

### Typed error (structured data)

```go
type ValidationError struct {
	Field string
	Reason string
}

func (e *ValidationError) Error() string { return "validation failed" }
```

Use when:
- you need field-level info
- you want to map to stable error codes

Consider wrapping a cause when you have a lower-level error:

```go
type DependencyError struct {
	Dependency string
	Err        error
}

func (e *DependencyError) Error() string { return "dependency failed: " + e.Dependency }
func (e *DependencyError) Unwrap() error { return e.Err }
```

## Good vs bad

Bad (breaks `errors.Is`):

```go
return fmt.Errorf("db: %v", err)
```

Good:

```go
return fmt.Errorf("db: %w", err)
```

Bad (double logging):

```go
log.Error("db failed", "err", err)
return err
```

Good (log once at boundary):

```go
return fmt.Errorf("db failed: %w", err)
```

## Mapping to HTTP

Keep mapping centralized (transport layer). Example mapping table:

- `ErrNotFound` → 404
- `*ValidationError` → 400 (or 422 if you standardize on it)
- uniqueness violation → 409
- dependency timeout → 503 (or 504 at gateway)
- unknown/unexpected → 500

Public response should include a stable `code` and safe `message` (see `http-api.md`).

## “Bad vs Good” (error branching)

```go
// ❌ BAD: branching by string matching (breaks on message changes)
if strings.Contains(err.Error(), "not found") { ... }

// ✅ GOOD: use errors.Is/As (stable)
if errors.Is(err, ErrNotFound) { ... }
var ve *ValidationError
if errors.As(err, &ve) { ... }
```

## nil interface trap (one of the nastiest Go error bugs)

```go
// ❌ BAD: returning a typed-nil as error makes it non-nil
func f() error {
	var e *MyError = nil
	return e // err != nil is true
}

// ✅ GOOD: return nil explicitly
func f() error {
	var e *MyError = nil
	if e == nil {
		return nil
	}
	return e
}
```
