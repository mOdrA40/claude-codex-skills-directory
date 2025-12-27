# Code Review Checklist

## Quick Scan (2 menit)

```
□ File > 500 lines? → Split
□ Function > 50 lines? → Extract
□ Nesting > 3 levels? → Flatten
□ `_ = someFunc()` untuk error? → BLOCKER
□ `panic()` di non-main? → BLOCKER
□ No tests untuk logic baru? → Request tests
```

## Correctness

```go
// ❌ Nil panic
name := user.Name  // user bisa nil

// ✅ Check first
if user == nil { return ErrNilUser }

// ❌ Index out of range
items[0]  // len bisa 0

// ✅ Check length
if len(items) > 0 { first := items[0] }
```

## Error Handling

```go
// ❌ BLOCKER - Error ignored
data, _ := json.Marshal(user)

// ✅ Handle semua error
data, err := json.Marshal(user)
if err != nil {
    return fmt.Errorf("marshal: %w", err)
}
```

## Security

```go
// ❌ SQL Injection
query := fmt.Sprintf("SELECT * FROM users WHERE id = %s", input)

// ✅ Parameterized
db.Query("SELECT * FROM users WHERE id = $1", input)

// ❌ Hardcoded secret
apiKey := "sk-live-xxx"

// ✅ Environment
apiKey := os.Getenv("API_KEY")
```

## Performance

```go
// ❌ N+1 queries
for _, u := range users {
    orders, _ := db.GetOrders(u.ID)  // N queries!
}

// ✅ Batch atau JOIN
orders := db.GetOrdersForUsers(userIDs)  // 1 query
```

## Testing

```go
// ❌ Unclear test name
func TestUser(t *testing.T)

// ✅ Descriptive
func TestCreateUser_WithValidInput_ReturnsUser(t *testing.T)
func TestCreateUser_WithDuplicateEmail_ReturnsError(t *testing.T)
```

## Response Templates

**Approval:**
```
LGTM! ✅
```

**Request Changes:**
```
Perlu revisi:
1. [issue] - [suggested fix]
```

**Blocker:**
```
⛔ BLOCKER: [issue]
Risk: [apa yang bisa terjadi]
Fix: [suggestion]
```
