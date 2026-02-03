# Design Patterns for Go

## Table of Contents
1. [Dependency Injection](#dependency-injection)
2. [Repository Pattern](#repository-pattern)
3. [Options Pattern](#options-pattern)
4. [Builder Pattern](#builder-pattern)
5. [Factory Pattern](#factory-pattern)
6. [Circuit Breaker](#circuit-breaker)
7. [Graceful Shutdown](#graceful-shutdown)
8. [Middleware Chain](#middleware-chain)

## Dependency Injection

### Interface-Based DI (RECOMMENDED)

```go
// domain/user.go - Define the interface at the consumer boundary
type UserRepository interface {
    GetByID(ctx context.Context, id string) (*User, error)
    Create(ctx context.Context, user *User) error
}

// usecase/user/service.go - Depend on interface
type UserService struct {
    repo UserRepository  // interface, not concrete
}

func NewUserService(repo UserRepository) *UserService {
    return &UserService{repo: repo}
}

func (s *UserService) GetUser(ctx context.Context, id string) (*User, error) {
    return s.repo.GetByID(ctx, id)
}

// repository/postgres/user.go - Implementation
type PostgresUserRepo struct {
    db *sqlx.DB
}

func NewPostgresUserRepo(db *sqlx.DB) *PostgresUserRepo {
    return &PostgresUserRepo{db: db}
}

func (r *PostgresUserRepo) GetByID(ctx context.Context, id string) (*User, error) {
    var user User
    err := r.db.GetContext(ctx, &user, "SELECT * FROM users WHERE id = $1", id)
    return &user, err
}

// cmd/api/main.go - Wire dependencies
func main() {
    db := setupDB()
    userRepo := postgres.NewPostgresUserRepo(db)
    userService := user.NewUserService(userRepo)  // Inject interface
    handler := http.NewUserHandler(userService)
}
```

## Repository Pattern

```go
// domain/user.go
type User struct {
    ID        string
    Name      string
    Email     string
    CreatedAt time.Time
}

type UserRepository interface {
    GetByID(ctx context.Context, id string) (*User, error)
    GetByEmail(ctx context.Context, email string) (*User, error)
    Create(ctx context.Context, user *User) error
    Update(ctx context.Context, user *User) error
    Delete(ctx context.Context, id string) error
    List(ctx context.Context, limit, offset int) ([]*User, error)
}

// repository/postgres/user.go
type userRepo struct {
    db *sqlx.DB
}

func NewUserRepository(db *sqlx.DB) UserRepository {
    return &userRepo{db: db}
}

func (r *userRepo) GetByID(ctx context.Context, id string) (*User, error) {
    var user User
    query := `SELECT id, name, email, created_at FROM users WHERE id = $1`
    if err := r.db.GetContext(ctx, &user, query, id); err != nil {
        if errors.Is(err, sql.ErrNoRows) {
            return nil, ErrNotFound
        }
        return nil, fmt.Errorf("get user by id: %w", err)
    }
    return &user, nil
}

func (r *userRepo) Create(ctx context.Context, user *User) error {
    query := `INSERT INTO users (id, name, email, created_at) VALUES ($1, $2, $3, $4)`
    _, err := r.db.ExecContext(ctx, query, user.ID, user.Name, user.Email, user.CreatedAt)
    if err != nil {
        return fmt.Errorf("create user: %w", err)
    }
    return nil
}
```

## Options Pattern

### Functional Options (RECOMMENDED)

```go
type Server struct {
    host         string
    port         int
    timeout      time.Duration
    maxConnections int
}

type Option func(*Server)

func WithHost(host string) Option {
    return func(s *Server) { s.host = host }
}

func WithPort(port int) Option {
    return func(s *Server) { s.port = port }
}

func WithTimeout(d time.Duration) Option {
    return func(s *Server) { s.timeout = d }
}

func WithMaxConnections(max int) Option {
    return func(s *Server) { s.maxConnections = max }
}

func NewServer(opts ...Option) *Server {
    // Default values
    s := &Server{
        host:         "localhost",
        port:         8080,
        timeout:      30 * time.Second,
        maxConnections: 100,
    }
    
    // Apply options
    for _, opt := range opts {
        opt(s)
    }
    
    return s
}

// Usage - clean and readable
server := NewServer(
    WithHost("0.0.0.0"),
    WithPort(9000),
    WithTimeout(60 * time.Second),
)
```

## Builder Pattern

```go
type QueryBuilder struct {
    table   string
    fields  []string
    where   []string
    orderBy string
    limit   int
}

func NewQueryBuilder(table string) *QueryBuilder {
    return &QueryBuilder{
        table:  table,
        fields: []string{"*"},
    }
}

func (qb *QueryBuilder) Select(fields ...string) *QueryBuilder {
    qb.fields = fields
    return qb
}

func (qb *QueryBuilder) Where(condition string) *QueryBuilder {
    qb.where = append(qb.where, condition)
    return qb
}

func (qb *QueryBuilder) OrderBy(field string) *QueryBuilder {
    qb.orderBy = field
    return qb
}

func (qb *QueryBuilder) Limit(n int) *QueryBuilder {
    qb.limit = n
    return qb
}

func (qb *QueryBuilder) Build() string {
    query := fmt.Sprintf("SELECT %s FROM %s", strings.Join(qb.fields, ", "), qb.table)
    
    if len(qb.where) > 0 {
        query += " WHERE " + strings.Join(qb.where, " AND ")
    }
    if qb.orderBy != "" {
        query += " ORDER BY " + qb.orderBy
    }
    if qb.limit > 0 {
        query += fmt.Sprintf(" LIMIT %d", qb.limit)
    }
    
    return query
}

// Usage
query := NewQueryBuilder("users").
    Select("id", "name", "email").
    Where("active = true").
    Where("role = 'admin'").
    OrderBy("created_at DESC").
    Limit(10).
    Build()
```

## Factory Pattern

```go
type NotificationSender interface {
    Send(ctx context.Context, to, message string) error
}

type EmailSender struct { /* ... */ }
type SMSSender struct { /* ... */ }
type PushSender struct { /* ... */ }

func (e *EmailSender) Send(ctx context.Context, to, message string) error { /* ... */ }
func (s *SMSSender) Send(ctx context.Context, to, message string) error { /* ... */ }
func (p *PushSender) Send(ctx context.Context, to, message string) error { /* ... */ }

type NotificationType string

const (
    NotificationEmail NotificationType = "email"
    NotificationSMS   NotificationType = "sms"
    NotificationPush  NotificationType = "push"
)

func NewNotificationSender(typ NotificationType, config Config) (NotificationSender, error) {
    switch typ {
    case NotificationEmail:
        return &EmailSender{config: config}, nil
    case NotificationSMS:
        return &SMSSender{config: config}, nil
    case NotificationPush:
        return &PushSender{config: config}, nil
    default:
        return nil, fmt.Errorf("unknown notification type: %s", typ)
    }
}
```

## Circuit Breaker

```go
type CircuitBreaker struct {
    mu           sync.Mutex
    failures     int
    successes    int
    state        State
    threshold    int
    timeout      time.Duration
    lastFailure  time.Time
}

type State int

const (
    StateClosed State = iota  // Normal operation
    StateOpen                  // Failing, reject calls
    StateHalfOpen             // Testing if service recovered
)

func NewCircuitBreaker(threshold int, timeout time.Duration) *CircuitBreaker {
    return &CircuitBreaker{
        threshold: threshold,
        timeout:   timeout,
        state:     StateClosed,
    }
}

func (cb *CircuitBreaker) Execute(fn func() error) error {
    cb.mu.Lock()
    defer cb.mu.Unlock()

    switch cb.state {
    case StateOpen:
        if time.Since(cb.lastFailure) > cb.timeout {
            cb.state = StateHalfOpen
        } else {
            return ErrCircuitOpen
        }
    }

    err := fn()
    
    if err != nil {
        cb.failures++
        cb.lastFailure = time.Now()
        if cb.failures >= cb.threshold {
            cb.state = StateOpen
        }
        return err
    }

    cb.successes++
    if cb.state == StateHalfOpen {
        cb.state = StateClosed
        cb.failures = 0
    }
    
    return nil
}

// Usage
cb := NewCircuitBreaker(5, 30*time.Second)
err := cb.Execute(func() error {
    return callExternalService()
})
```

## Graceful Shutdown

```go
func main() {
    ctx, cancel := context.WithCancel(context.Background())
    
    // Setup signal handling
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
    
    // Start server
    server := &http.Server{Addr: ":8080", Handler: router}
    go func() {
        if err := server.ListenAndServe(); err != http.ErrServerClosed {
            log.Fatal(err)
        }
    }()
    
    // Start workers
    var wg sync.WaitGroup
    for i := 0; i < 5; i++ {
        wg.Add(1)
        go worker(ctx, &wg)
    }
    
    // Wait for signal
    <-sigChan
    log.Println("Shutdown signal received")
    cancel()  // Cancel context
    
    // Shutdown server with timeout
    shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer shutdownCancel()
    
    if err := server.Shutdown(shutdownCtx); err != nil {
        log.Printf("Server shutdown error: %v", err)
    }
    
    // Wait for workers
    wg.Wait()
    log.Println("Graceful shutdown complete")
}

func worker(ctx context.Context, wg *sync.WaitGroup) {
    defer wg.Done()
    for {
        select {
        case <-ctx.Done():
            log.Println("Worker shutting down")
            return
        default:
            doWork()
        }
    }
}
```

## Middleware Chain

```go
type Middleware func(http.Handler) http.Handler

func Chain(middlewares ...Middleware) Middleware {
    return func(final http.Handler) http.Handler {
        for i := len(middlewares) - 1; i >= 0; i-- {
            final = middlewares[i](final)
        }
        return final
    }
}

// Logging middleware
func Logger(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        next.ServeHTTP(w, r)
        log.Printf("%s %s %v", r.Method, r.URL.Path, time.Since(start))
    })
}

// Auth middleware
func Auth(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        token := r.Header.Get("Authorization")
        if token == "" {
            http.Error(w, "Unauthorized", http.StatusUnauthorized)
            return
        }
        next.ServeHTTP(w, r)
    })
}

// Recovery middleware
func Recovery(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if err := recover(); err != nil {
                log.Printf("Panic recovered: %v", err)
                http.Error(w, "Internal Server Error", http.StatusInternalServerError)
            }
        }()
        next.ServeHTTP(w, r)
    })
}

// Usage
chain := Chain(Recovery, Logger, Auth)
http.Handle("/api/", chain(apiHandler))
```

## Anti-Patterns to Avoid

| Anti-Pattern | Problem | Better Approach |
|--------------|---------|-----------------|
| God Object | One struct does everything | Split into focused structs |
| Singleton | Hard to test, hidden deps | Dependency Injection |
| Service Locator | Hidden dependencies | Explicit constructor injection |
| Premature Abstraction | Over-engineering | YAGNI - abstract when needed |
| Deep Inheritance | Go has no inheritance | Composition over inheritance |
