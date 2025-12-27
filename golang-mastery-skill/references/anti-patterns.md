# Anti-Patterns & Bug Traps

Pola-pola yang sering menyebabkan bug di production.

## Concurrency

### Loop Variable Capture

```go
// ❌ Semua goroutine dapat nilai terakhir
for _, item := range items {
    go func() {
        process(item)  // item = nilai terakhir!
    }()
}

// ✅ Pass as parameter
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

// ✅ Use sync.RWMutex atau sync.Map
var cache sync.Map
cache.Store("key", "value")
```

### Goroutine Leak

```go
// ❌ Blocks forever
ch := make(chan int)
go func() {
    ch <- 1  // No receiver = leak
}()

// ✅ Buffered atau context
ch := make(chan int, 1)
```

### Defer in Loop

```go
// ❌ Files tidak ditutup sampai return
for _, path := range paths {
    f, _ := os.Open(path)
    defer f.Close()  // All stacked!
}

// ✅ Extract ke function
for _, path := range paths {
    processFile(path)
}
func processFile(path string) {
    f, _ := os.Open(path)
    defer f.Close()
}
```

## Memory

### Slice Aliasing

```go
// ❌ Modifies original
func getFirst3(data []int) []int {
    return data[:3]  // Shares array!
}

// ✅ Copy
func getFirst3(data []int) []int {
    result := make([]int, 3)
    copy(result, data[:3])
    return result
}
```

## Error Handling

### Shadow Error

```go
// ❌ Outer err tidak di-set
var err error
if condition {
    result, err := doSomething()  // Shadows!
}
return err  // Selalu nil

// ✅ Assign to existing
var err error
if condition {
    var result Result
    result, err = doSomething()
}
```

### nil Interface Trap

```go
// ❌ Returns non-nil interface
func process() error {
    var err *MyError = nil
    return err  // err != nil is TRUE!
}

// ✅ Return nil explicitly
if err == nil {
    return nil
}
```

## HTTP

### Body Not Closed

```go
// ❌ Connection leak
resp, _ := http.Get(url)
// Forgot resp.Body.Close()

// ✅ Always close
resp, _ := http.Get(url)
defer resp.Body.Close()
io.Copy(io.Discard, resp.Body)  // Drain for reuse
```

### No Timeout

```go
// ❌ Hangs forever
http.Get(url)

// ✅ Custom client
client := &http.Client{Timeout: 30 * time.Second}
```

## Database

### Rows Not Closed

```go
// ❌ Connection exhausted
rows, _ := db.Query("SELECT ...")
// Forgot rows.Close()

// ✅ Always close
rows, _ := db.Query("SELECT ...")
defer rows.Close()
for rows.Next() { }
rows.Err()  // Check error
```

## Time

### time.After Leak

```go
// ❌ Timer not GC'd in loop
for {
    select {
    case <-time.After(time.Second):
    }
}

// ✅ Reuse timer
timer := time.NewTimer(time.Second)
for {
    select {
    case <-timer.C:
        timer.Reset(time.Second)
    }
}
```
