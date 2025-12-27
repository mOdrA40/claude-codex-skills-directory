# Coding Standards

Standar wajib yang dipaksakan di code review.

## Error Handling

```go
// ❌ FATAL
data, _ := json.Marshal(user)

// ✅ BENAR
data, err := json.Marshal(user)
if err != nil {
    return fmt.Errorf("marshal user %d: %w", user.ID, err)
}
```

## Naming

```go
// Variables
for i, v := range items { }        // Short untuk scope kecil
userRepository := NewUserRepo(db)  // Descriptive untuk scope besar

// Functions - Verb + Noun
func CreateUser(ctx context.Context, req CreateUserRequest) (*User, error)
func (u *User) Name() string       // Getter tanpa "Get"

// Acronyms - All caps atau all lower
userID, UserID, httpClient, HTTPClient
```

## Function Design

```go
// Max 4 params - gunakan struct jika lebih
type CreateOrderParams struct {
    UserID    int64
    ProductID int64
    Quantity  int
}

// Max 50 lines - decompose jika lebih
// Max 3 level nesting - gunakan early return
```

## Early Return

```go
// ❌ Deep nesting
if user != nil {
    if user.IsActive() {
        // logic
    }
}

// ✅ Guard clauses
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
// ✅ Small interfaces
type Reader interface {
    Read(p []byte) (n int, err error)
}

// ✅ Define at consumer, not producer
// ✅ Accept interface, return concrete
```

## Concurrency

```go
// ✅ Always use context
func (s *Service) LongOp(ctx context.Context) error {
    select {
    case <-ctx.Done():
        return ctx.Err()
    case result := <-s.doWork():
        return s.process(result)
    }
}
```

## Comments

```go
// ❌ Useless
// increment counter
counter++

// ✅ Explain WHY
// 30s timeout: auth service p99 latency 25s under load
ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
```
