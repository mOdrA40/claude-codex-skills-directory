# Code Review Checklist

## Quick Scan (2 minutes)

```
□ File > 500 lines? → Split
□ Function > 50 lines? → Extract
□ Nesting > 3 levels? → Flatten
□ `_ = someFunc()` on an error-returning call? → BLOCKER
□ `panic()` in non-entrypoint code? → BLOCKER (unless truly impossible invariant)
□ No tests for new logic? → Request tests
```

## Correctness

```go
// ❌ BAD: nil panic
name := user.Name // user may be nil

// ✅ GOOD: check first
if user == nil { return ErrNilUser }

// ❌ BAD: index out of range
items[0] // len may be 0

// ✅ GOOD: check length
if len(items) > 0 { first := items[0] }
```

## Error Handling

```go
// ❌ BLOCKER: error ignored
data, _ := json.Marshal(user)

// ✅ GOOD: handle all errors
data, err := json.Marshal(user)
if err != nil {
    return fmt.Errorf("marshal: %w", err)
}
```

## Security

```go
// ❌ BAD: SQL injection
query := fmt.Sprintf("SELECT * FROM users WHERE id = %s", input)

// ✅ GOOD: parameterized query
db.Query("SELECT * FROM users WHERE id = $1", input)

// ❌ BAD: hardcoded secret
apiKey := "sk-live-xxx"

// ✅ GOOD: environment / secret manager
apiKey := os.Getenv("API_KEY")
```

## Performance

```go
// ❌ BAD: N+1 queries
for _, u := range users {
    orders, _ := db.GetOrders(u.ID)  // N queries!
}

// ✅ GOOD: batch or JOIN
orders := db.GetOrdersForUsers(userIDs)  // 1 query
```

## Testing

```go
// ❌ BAD: unclear test name
func TestUser(t *testing.T)

// ✅ GOOD: descriptive
func TestCreateUser_WithValidInput_ReturnsUser(t *testing.T)
func TestCreateUser_WithDuplicateEmail_ReturnsError(t *testing.T)
```

## Concurrency (common production blockers)

```go
// ❌ BAD: goroutine leak (no stop condition)
go func() {
    for v := range ch {
        process(v)
    }
}()

// ✅ GOOD: explicit ownership and shutdown via context
go func() {
    for {
        select {
        case <-ctx.Done():
            return
        case v, ok := <-ch:
            if !ok { return }
            process(v)
        }
    }
}()
```

## Timeouts (outbound IO)

```go
// ❌ BAD: can hang forever
resp, err := http.Get(url)

// ✅ GOOD: request-scoped context + hard client timeout
client := &http.Client{Timeout: 10 * time.Second}
req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
resp, err := client.Do(req)
```

## Response Templates

**Approval:**
```
LGTM! ✅
```

**Request Changes:**
```
Please revise:
1. [issue] - [suggested fix]
```

**Blocker:**
```
⛔ BLOCKER: [issue]
Risk: [what can go wrong in production]
Fix: [suggestion]
```
