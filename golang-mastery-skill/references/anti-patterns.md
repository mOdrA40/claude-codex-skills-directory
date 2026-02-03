# Anti-Patterns & Bug Traps

Patterns that frequently cause production bugs. Prefer these “guardrails” during code review.

## Concurrency

### Loop Variable Capture

```go
// ❌ BAD: all goroutines see the last value
for _, item := range items {
    go func() {
        process(item) // item = last value!
    }()
}

// ✅ GOOD: pass as parameter
for _, item := range items {
    go func(i Item) {
        process(i)
    }(item)
}
```

### Concurrent Map

```go
// ❌ fatal: concurrent map writes
var cache = make(map[string]string)
go func() { cache["a"] = "1" }()
go func() { cache["b"] = "2" }()

// ✅ GOOD: use sync.RWMutex or sync.Map
var cache sync.Map
cache.Store("key", "value")
```

### Goroutine Leak

```go
// ❌ BAD: blocks forever (leak)
ch := make(chan int)
go func() {
    ch <- 1 // no receiver = leak
}()

// ✅ GOOD: buffered channel OR context-driven shutdown
ch := make(chan int, 1)
```

### Defer in Loop

```go
// ❌ BAD: files won't close until the outer function returns
for _, path := range paths {
    f, _ := os.Open(path)
    defer f.Close() // all stacked!
}

// ✅ GOOD: move into a helper function
for _, path := range paths {
    processFile(path)
}
func processFile(path string) {
    f, _ := os.Open(path)
    defer f.Close()
}
```

### Ignoring Context in Loops

```go
// ❌ BAD: ignores cancellation, keeps burning CPU/IO
for {
    doWork()
}

// ✅ GOOD: stop condition is explicit
for {
    select {
    case <-ctx.Done():
        return ctx.Err()
    default:
        doWork()
    }
}
```

## Memory

### Slice Aliasing

```go
// ❌ Modifies original
func getFirst3(data []int) []int {
    return data[:3]  // Shares array!
}

// ✅ GOOD: copy
func getFirst3(data []int) []int {
    result := make([]int, 3)
    copy(result, data[:3])
    return result
}
```

## Error Handling

### Shadow Error

```go
// ❌ BAD: outer err never updated
var err error
if condition {
    result, err := doSomething() // shadows!
}
return err // always nil

// ✅ GOOD: assign to existing variable
var err error
if condition {
    var result Result
    result, err = doSomething()
}
```

### nil Interface Trap

```go
// ❌ BAD: returns non-nil interface
func process() error {
    var err *MyError = nil
    return err // err != nil is TRUE!
}

// ✅ GOOD: return nil explicitly
if err == nil {
    return nil
}
```

## HTTP

### Body Not Closed

```go
// ❌ BAD: connection leak
resp, _ := http.Get(url)
// Forgot resp.Body.Close()

// ✅ GOOD: always close and drain
resp, _ := http.Get(url)
defer resp.Body.Close()
io.Copy(io.Discard, resp.Body)  // Drain for reuse
```

### No Timeout

```go
// ❌ Hangs forever
http.Get(url)

// ✅ GOOD: custom client with timeout
client := &http.Client{Timeout: 30 * time.Second}
```

### Mutating `http.DefaultTransport`

```go
// ❌ BAD: global side-effect; surprises other packages/tests
http.DefaultTransport.(*http.Transport).MaxIdleConnsPerHost = 200

// ✅ GOOD: clone and own your transport
base := http.DefaultTransport.(*http.Transport).Clone()
base.MaxIdleConnsPerHost = 200
client := &http.Client{Transport: base}
```

## Database

### Rows Not Closed

```go
// ❌ Connection exhausted
rows, _ := db.Query("SELECT ...")
// Forgot rows.Close()

// ✅ GOOD: always close and check rows.Err
rows, _ := db.Query("SELECT ...")
defer rows.Close()
for rows.Next() { }
rows.Err()  // Check error
```

## Time

### time.After Leak

```go
// ❌ BAD: allocates a timer every loop iteration
for {
    select {
    case <-time.After(time.Second):
    }
}

// ✅ GOOD: reuse timer
timer := time.NewTimer(time.Second)
for {
    select {
    case <-timer.C:
        timer.Reset(time.Second)
    }
}
```
