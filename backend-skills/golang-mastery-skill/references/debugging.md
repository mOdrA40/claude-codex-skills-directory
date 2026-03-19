# Advanced Debugging Techniques

## Table of Contents
1. [Profiling](#profiling)
2. [Race Detection](#race-detection)
3. [Memory Analysis](#memory-analysis)
4. [Deadlock Detection](#deadlock-detection)
5. [Delve Debugger](#delve-debugger)
6. [Production Debugging](#production-debugging)
7. [Common Bug Patterns](#common-bug-patterns)

## Profiling

### CPU Profiling

```go
import _ "net/http/pprof"
go http.ListenAndServe(":6060", nil)
```

```bash
# Generate profile
go test -cpuprofile cpu.prof -bench .

# Analyze
go tool pprof cpu.prof
(pprof) top10
(pprof) list funcName
(pprof) web

# Web UI
go tool pprof -http=:8080 cpu.prof
```

### Memory Profiling

```bash
go test -memprofile mem.prof -bench .
go tool pprof -alloc_space mem.prof
go tool pprof -inuse_space mem.prof
```

### Trace Analysis

```bash
go test -trace trace.out
go tool trace trace.out
```

## Race Detection

```bash
# Strongly recommended before merging concurrency changes
go test -race ./...
go build -race ./cmd/api
```

### Common Race Patterns

```go
// ❌ BUG: Loop variable capture
for i := 0; i < 10; i++ {
    go func() { fmt.Println(i) }()  // RACE
}

// ✅ FIX
for i := 0; i < 10; i++ {
    go func(n int) { fmt.Println(n) }(i)
}

// ❌ BUG: Shared slice
var results []int
for _, item := range items {
    go func(item Item) {
        results = append(results, process(item))  // RACE!
    }(item)
}

// ✅ FIX: Use channel
resultsChan := make(chan int, len(items))
for _, item := range items {
    go func(item Item) {
        resultsChan <- process(item)
    }(item)
}
```

## Memory Analysis

### Finding Memory Leaks

```go
import "runtime"

var m runtime.MemStats
runtime.ReadMemStats(&m)
fmt.Printf("Alloc = %v MiB\n", m.Alloc/1024/1024)
```

### Common Leak Patterns

```go
// ❌ LEAK: goroutine never exits
go func() {
    for { data := <-ch; process(data) }  // Stuck forever
}()

// ✅ FIX: Context cancellation
go func() {
    for {
        select {
        case <-ctx.Done(): return
        case data := <-ch: process(data)
        }
    }
}()

// ❌ LEAK: Ticker not stopped
ticker := time.NewTicker(time.Second)

// ✅ FIX
ticker := time.NewTicker(time.Second)
defer ticker.Stop()

// ❌ LEAK: HTTP body not closed
resp, _ := http.Get(url)

// ✅ FIX
resp, err := http.Get(url)
if err != nil { return err }
defer resp.Body.Close()
```

## Deadlock Detection

```bash
GODEBUG=schedtrace=1000 ./app
GODEBUG=scheddetail=1,schedtrace=1000 ./app
```

### Deadlock Patterns

```go
// ❌ DEADLOCK: Send while holding lock
s.mu.Lock()
s.ch <- data  // DEADLOCK
s.mu.Unlock()

// ✅ FIX
s.mu.Lock()
data := s.prepareData()
s.mu.Unlock()
s.ch <- data

// ❌ DEADLOCK: Inconsistent lock order
a.mu.Lock()
b.mu.Lock()

// ✅ FIX: Order by ID
first, second := a, b
if a.ID > b.ID { first, second = b, a }
first.mu.Lock()
second.mu.Lock()
```

## Delve Debugger

```bash
go install github.com/go-delve/delve/cmd/dlv@latest

dlv debug ./cmd/api
dlv test ./pkg/mypackage
dlv attach <pid>
```

```
(dlv) break main.go:42
(dlv) continue
(dlv) next
(dlv) step
(dlv) print x
(dlv) locals
(dlv) goroutines
(dlv) stack
```

## Production Debugging

### Safety note

Never expose `net/http/pprof` to the public internet. If you enable pprof in production:
- bind to localhost, or
- protect it with auth / network policy, or
- serve it on an admin port not reachable from the outside.

### pprof Endpoints

```bash
go tool pprof http://localhost:6060/debug/pprof/profile?seconds=30
go tool pprof http://localhost:6060/debug/pprof/heap
curl http://localhost:6060/debug/pprof/goroutine?debug=2
```

### Health Check

```go
func healthHandler(w http.ResponseWriter, r *http.Request) {
    json.NewEncoder(w).Encode(map[string]interface{}{
        "status":     "ok",
        "goroutines": runtime.NumGoroutine(),
        "memory_mb":  m.Alloc / 1024 / 1024,
    })
}
```

## Common Bug Patterns

| Bug | Pattern | Fix |
|-----|---------|-----|
| Nil pointer | `user.Name` without a nil check | `if user != nil` |
| Index out of range | `items[5]` without bounds check | `if len(items) > 5` |
| Close nil channel | `close(ch)` without a nil check | ensure channel is initialized (prefer ownership rules; avoid closing nil) |
| Concurrent map | `m["a"] = 1` concurrently | `sync.Map` or a mutex |
| Defer arg eval | `defer fmt.Println(i)` | `defer func(n int){...}(i)` |

## Quick Debug Checklist

1. ✅ `go test -race ./...`
2. ✅ `go vet ./...`
3. ✅ `golangci-lint run`
4. ✅ Error handling complete?
5. ✅ Nil checks before dereference?
6. ✅ Context cancellation for goroutines?
7. ✅ defer Close() for resources?

## “Bad vs Good” (debuggability)

```go
// ❌ BAD: swallowing root cause
if err != nil {
    return errors.New("failed to save")
}

// ✅ GOOD: keep the cause while adding context
if err != nil {
    return fmt.Errorf("save user: %w", err)
}
```
