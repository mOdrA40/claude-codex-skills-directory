# Coding Standards

Review-enforced standards for Go codebases. These are intentionally opinionated and optimized for production systems (readability, reliability, operability).

## Error Handling

```go
// ❌ BAD: error ignored
data, _ := json.Marshal(user)

// ✅ GOOD: wrap with context (preserves errors.Is/As via %w)
data, err := json.Marshal(user)
if err != nil {
    return fmt.Errorf("marshal user %d: %w", user.ID, err)
}
```

## Naming

```go
// Variables
for i, v := range items { }        // short is OK for tiny scope
userRepository := NewUserRepo(db)  // descriptive for long scope / shared state

// Functions - Verb + Noun
func CreateUser(ctx context.Context, req CreateUserRequest) (*User, error)
func (u *User) Name() string       // getters without "Get"

// Acronyms - consistent casing
userID, UserID, httpClient, HTTPClient
```

## Function Design

```go
// Prefer <= 4 params. Use a params struct when you have more.
type CreateOrderParams struct {
    UserID    int64
    ProductID int64
    Quantity  int
}

// Keep functions small:
// - Prefer <= ~50 lines
// - Prefer <= 3 nesting levels (use guard clauses / early returns)
```

## Early Return

```go
// ❌ BAD: deep nesting
if user != nil {
    if user.IsActive() {
        // logic
    }
}

// ✅ GOOD: guard clauses
if user == nil {
    return ErrNilUser
}
if !user.IsActive() {
    return ErrInactiveUser
}
// logic
```

## Struct Design

```go
type User struct {
    // 1. Identity
    ID   int64
    UUID string
    
    // 2. Core fields
    Email string
    Name  string
    
    // 3. State
    IsActive bool
    
    // 4. Timestamps
    CreatedAt time.Time
    UpdatedAt time.Time
}
```

## Interface

```go
// ✅ GOOD: small interfaces
type Reader interface {
    Read(p []byte) (n int, err error)
}

// ✅ Define interfaces at the consumer boundary (not the producer).
// ✅ Accept interfaces, return concretes.
```

## Concurrency

```go
// ✅ GOOD: honor context in loops and IO
func (s *Service) LongOp(ctx context.Context) error {
    select {
    case <-ctx.Done():
        return ctx.Err()
    case result := <-s.doWork():
        return s.process(result)
    }
}
```

## Context

```go
// ❌ BAD: losing cancellation and deadlines
func (s *Service) Handle(ctx context.Context, req Request) error {
    return s.repo.Save(context.Background(), req) // breaks cancellation
}

// ✅ GOOD: pass the inbound ctx to downstream calls
func (s *Service) Handle(ctx context.Context, req Request) error {
    return s.repo.Save(ctx, req)
}
```

## Logging

```go
// ❌ BAD: double-logging (log here AND return error to be logged again)
if err != nil {
    slog.Error("db failed", "err", err)
    return err
}

// ✅ GOOD: wrap, return; log once at a boundary (HTTP/CLI/job runner)
if err != nil {
    return fmt.Errorf("insert user: %w", err)
}
```

## Tests (determinism)

```go
// ❌ BAD: time.Now makes tests flaky
expiresAt := time.Now().Add(10 * time.Minute)

// ✅ GOOD: inject clock (or pass time in)
type Clock interface{ Now() time.Time }
expiresAt := clock.Now().Add(10 * time.Minute)
```

## Comments

```go
// ❌ BAD: says what the code already says
// increment counter
counter++

// ✅ GOOD: explain WHY (tradeoff / production constraint)
// 30s timeout: auth service p99 latency ~25s under peak load
ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
```
