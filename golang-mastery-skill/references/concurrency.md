# Concurrency Patterns

## Table of Contents
1. [Mutex Patterns](#mutex-patterns)
2. [Channel Patterns](#channel-patterns)
3. [Worker Pool](#worker-pool)
4. [Context Cancellation](#context-cancellation)
5. [Race Condition Prevention](#race-condition-prevention)

## Mutex Patterns

### Basic Mutex

```go
type SafeCounter struct {
    mu    sync.Mutex
    count int
}

func (c *SafeCounter) Increment() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.count++
}
```

### RWMutex (Read-Heavy)

```go
type Cache struct {
    mu   sync.RWMutex
    data map[string]interface{}
}

func (c *Cache) Get(key string) (interface{}, bool) {
    c.mu.RLock()
    defer c.mu.RUnlock()
    v, ok := c.data[key]
    return v, ok
}

func (c *Cache) Set(key string, value interface{}) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.data[key] = value
}
```

### Lock Ordering (Prevent Deadlock)

```go
// ❌ DEADLOCK RISK
func transfer(from, to *Account, amount int) {
    from.mu.Lock()
    to.mu.Lock()  // Deadlock jika lock sebaliknya
}

// ✅ SAFE - consistent order by ID
func transfer(from, to *Account, amount int) {
    first, second := from, to
    if from.ID > to.ID {
        first, second = to, from
    }
    first.mu.Lock()
    defer first.mu.Unlock()
    second.mu.Lock()
    defer second.mu.Unlock()
}
```

## Channel Patterns

### Fan-Out/Fan-In

```go
func fanOut(input <-chan int, workers int) []<-chan int {
    channels := make([]<-chan int, workers)
    for i := 0; i < workers; i++ {
        channels[i] = worker(input)
    }
    return channels
}

func fanIn(channels ...<-chan int) <-chan int {
    var wg sync.WaitGroup
    out := make(chan int)

    output := func(c <-chan int) {
        defer wg.Done()
        for n := range c {
            out <- n
        }
    }

    wg.Add(len(channels))
    for _, c := range channels {
        go output(c)
    }

    go func() {
        wg.Wait()
        close(out)
    }()

    return out
}
```

### Semaphore

```go
type Semaphore struct {
    sem chan struct{}
}

func NewSemaphore(max int) *Semaphore {
    return &Semaphore{sem: make(chan struct{}, max)}
}

func (s *Semaphore) Acquire() { s.sem <- struct{}{} }
func (s *Semaphore) Release() { <-s.sem }

// Usage
sem := NewSemaphore(10)
for _, job := range jobs {
    sem.Acquire()
    go func(j Job) {
        defer sem.Release()
        process(j)
    }(job)
}
```

## Worker Pool

```go
type WorkerPool struct {
    jobs    chan Job
    results chan Result
    wg      sync.WaitGroup
}

func NewWorkerPool(numWorkers int) *WorkerPool {
    pool := &WorkerPool{
        jobs:    make(chan Job, 100),
        results: make(chan Result, 100),
    }
    for i := 0; i < numWorkers; i++ {
        pool.wg.Add(1)
        go pool.worker()
    }
    return pool
}

func (p *WorkerPool) worker() {
    defer p.wg.Done()
    for job := range p.jobs {
        p.results <- process(job)
    }
}

func (p *WorkerPool) Close() {
    close(p.jobs)
    p.wg.Wait()
    close(p.results)
}
```

## Context Cancellation

### Graceful Shutdown

```go
func main() {
    ctx, cancel := context.WithCancel(context.Background())
    
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
    
    go func() {
        <-sigChan
        cancel()
    }()
    
    var wg sync.WaitGroup
    for i := 0; i < 5; i++ {
        wg.Add(1)
        go worker(ctx, &wg)
    }
    
    wg.Wait()
}

func worker(ctx context.Context, wg *sync.WaitGroup) {
    defer wg.Done()
    for {
        select {
        case <-ctx.Done():
            return
        default:
            doWork()
        }
    }
}
```

### Timeout Pattern

```go
func fetchWithTimeout(ctx context.Context, url string) ([]byte, error) {
    ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
    defer cancel()

    req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    return io.ReadAll(resp.Body)
}
```

## Race Condition Prevention

### Common Bugs

```go
// ❌ Loop variable capture
for i := 0; i < 10; i++ {
    go func() { fmt.Println(i) }()  // RACE
}

// ✅ Pass as argument
for i := 0; i < 10; i++ {
    go func(n int) { fmt.Println(n) }(i)
}

// ❌ Concurrent map access
m := make(map[string]int)
go func() { m["a"] = 1 }()  // CRASH

// ✅ Use sync.Map
var m sync.Map
go func() { m.Store("a", 1) }()
```

### Detection

```bash
go test -race ./...
go build -race ./cmd/api
```
