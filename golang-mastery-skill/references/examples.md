# Real-World Code Examples

Contoh kode production-ready yang bisa langsung digunakan.

## Table of Contents
1. [HTTP Server Skeleton](#http-server-skeleton)
2. [Repository Pattern](#repository-pattern)
3. [Service Layer](#service-layer)
4. [Middleware Stack](#middleware-stack)
5. [Worker/Consumer](#workerconsumer)
6. [Graceful Shutdown](#graceful-shutdown)
7. [Config Management](#config-management)
8. [Custom Error Types](#custom-error-types)

---

## HTTP Server Skeleton

```go
// cmd/api/main.go
package main

import (
    "context"
    "log/slog"
    "net/http"
    "os"
    "os/signal"
    "syscall"
    "time"
    
    "github.com/go-chi/chi/v5"
    "github.com/go-chi/chi/v5/middleware"
)

func main() {
    // Setup logger
    logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
    slog.SetDefault(logger)
    
    // Load config
    cfg, err := LoadConfig()
    if err != nil {
        slog.Error("failed to load config", "error", err)
        os.Exit(1)
    }
    
    // Setup dependencies
    db, err := NewDatabase(cfg.DatabaseURL)
    if err != nil {
        slog.Error("failed to connect database", "error", err)
        os.Exit(1)
    }
    defer db.Close()
    
    // Setup router
    r := chi.NewRouter()
    
    // Global middleware
    r.Use(middleware.RequestID)
    r.Use(middleware.RealIP)
    r.Use(middleware.Logger)
    r.Use(middleware.Recoverer)
    r.Use(middleware.Timeout(60 * time.Second))
    
    // Health check
    r.Get("/health", func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
        w.Write([]byte("OK"))
    })
    
    // API routes
    r.Route("/api/v1", func(r chi.Router) {
        // User routes
        userRepo := NewUserRepository(db)
        userSvc := NewUserService(userRepo)
        userHandler := NewUserHandler(userSvc)
        
        r.Route("/users", func(r chi.Router) {
            r.Get("/", userHandler.List)
            r.Post("/", userHandler.Create)
            r.Get("/{id}", userHandler.Get)
            r.Put("/{id}", userHandler.Update)
            r.Delete("/{id}", userHandler.Delete)
        })
    })
    
    // Create server
    srv := &http.Server{
        Addr:         cfg.ServerAddr,
        Handler:      r,
        ReadTimeout:  15 * time.Second,
        WriteTimeout: 15 * time.Second,
        IdleTimeout:  60 * time.Second,
    }
    
    // Start server
    go func() {
        slog.Info("starting server", "addr", cfg.ServerAddr)
        if err := srv.ListenAndServe(); err != http.ErrServerClosed {
            slog.Error("server error", "error", err)
            os.Exit(1)
        }
    }()
    
    // Graceful shutdown
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
    <-quit
    
    slog.Info("shutting down server...")
    
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    
    if err := srv.Shutdown(ctx); err != nil {
        slog.Error("forced shutdown", "error", err)
    }
    
    slog.Info("server stopped")
}
```

---

## Repository Pattern

```go
// internal/repository/user_repository.go
package repository

import (
    "context"
    "database/sql"
    "errors"
    "fmt"
    
    "github.com/jmoiron/sqlx"
)

var ErrNotFound = errors.New("record not found")

type User struct {
    ID        int64     `db:"id"`
    Email     string    `db:"email"`
    Name      string    `db:"name"`
    CreatedAt time.Time `db:"created_at"`
    UpdatedAt time.Time `db:"updated_at"`
}

type UserRepository struct {
    db *sqlx.DB
}

func NewUserRepository(db *sqlx.DB) *UserRepository {
    return &UserRepository{db: db}
}

func (r *UserRepository) FindByID(ctx context.Context, id int64) (*User, error) {
    var user User
    query := `SELECT id, email, name, created_at, updated_at 
              FROM users WHERE id = $1`
    
    if err := r.db.GetContext(ctx, &user, query, id); err != nil {
        if errors.Is(err, sql.ErrNoRows) {
            return nil, ErrNotFound
        }
        return nil, fmt.Errorf("find user %d: %w", id, err)
    }
    return &user, nil
}

func (r *UserRepository) FindByEmail(ctx context.Context, email string) (*User, error) {
    var user User
    query := `SELECT id, email, name, created_at, updated_at 
              FROM users WHERE email = $1`
    
    if err := r.db.GetContext(ctx, &user, query, email); err != nil {
        if errors.Is(err, sql.ErrNoRows) {
            return nil, ErrNotFound
        }
        return nil, fmt.Errorf("find user by email: %w", err)
    }
    return &user, nil
}

func (r *UserRepository) Create(ctx context.Context, user *User) error {
    query := `INSERT INTO users (email, name, created_at, updated_at) 
              VALUES ($1, $2, $3, $4) RETURNING id`
    
    now := time.Now().UTC()
    user.CreatedAt = now
    user.UpdatedAt = now
    
    if err := r.db.QueryRowContext(ctx, query, 
        user.Email, user.Name, user.CreatedAt, user.UpdatedAt,
    ).Scan(&user.ID); err != nil {
        return fmt.Errorf("create user: %w", err)
    }
    return nil
}

func (r *UserRepository) Update(ctx context.Context, user *User) error {
    query := `UPDATE users SET email = $1, name = $2, updated_at = $3 
              WHERE id = $4`
    
    user.UpdatedAt = time.Now().UTC()
    
    result, err := r.db.ExecContext(ctx, query, 
        user.Email, user.Name, user.UpdatedAt, user.ID)
    if err != nil {
        return fmt.Errorf("update user: %w", err)
    }
    
    rows, _ := result.RowsAffected()
    if rows == 0 {
        return ErrNotFound
    }
    return nil
}

func (r *UserRepository) Delete(ctx context.Context, id int64) error {
    query := `DELETE FROM users WHERE id = $1`
    
    result, err := r.db.ExecContext(ctx, query, id)
    if err != nil {
        return fmt.Errorf("delete user: %w", err)
    }
    
    rows, _ := result.RowsAffected()
    if rows == 0 {
        return ErrNotFound
    }
    return nil
}

func (r *UserRepository) List(ctx context.Context, limit, offset int) ([]*User, error) {
    var users []*User
    query := `SELECT id, email, name, created_at, updated_at 
              FROM users ORDER BY id LIMIT $1 OFFSET $2`
    
    if err := r.db.SelectContext(ctx, &users, query, limit, offset); err != nil {
        return nil, fmt.Errorf("list users: %w", err)
    }
    return users, nil
}
```

---

## Service Layer

```go
// internal/service/user_service.go
package service

import (
    "context"
    "errors"
    "fmt"
    
    "myapp/internal/repository"
)

var (
    ErrUserNotFound    = errors.New("user not found")
    ErrDuplicateEmail  = errors.New("email already exists")
    ErrInvalidInput    = errors.New("invalid input")
)

type UserRepository interface {
    FindByID(ctx context.Context, id int64) (*repository.User, error)
    FindByEmail(ctx context.Context, email string) (*repository.User, error)
    Create(ctx context.Context, user *repository.User) error
    Update(ctx context.Context, user *repository.User) error
    Delete(ctx context.Context, id int64) error
    List(ctx context.Context, limit, offset int) ([]*repository.User, error)
}

type UserService struct {
    repo UserRepository
}

func NewUserService(repo UserRepository) *UserService {
    return &UserService{repo: repo}
}

type CreateUserInput struct {
    Email string
    Name  string
}

func (s *UserService) CreateUser(ctx context.Context, input CreateUserInput) (*repository.User, error) {
    // Validate
    if input.Email == "" || input.Name == "" {
        return nil, ErrInvalidInput
    }
    
    // Check duplicate
    existing, err := s.repo.FindByEmail(ctx, input.Email)
    if err != nil && !errors.Is(err, repository.ErrNotFound) {
        return nil, fmt.Errorf("check existing email: %w", err)
    }
    if existing != nil {
        return nil, ErrDuplicateEmail
    }
    
    // Create
    user := &repository.User{
        Email: input.Email,
        Name:  input.Name,
    }
    
    if err := s.repo.Create(ctx, user); err != nil {
        return nil, fmt.Errorf("create user: %w", err)
    }
    
    return user, nil
}

func (s *UserService) GetUser(ctx context.Context, id int64) (*repository.User, error) {
    user, err := s.repo.FindByID(ctx, id)
    if err != nil {
        if errors.Is(err, repository.ErrNotFound) {
            return nil, ErrUserNotFound
        }
        return nil, fmt.Errorf("get user: %w", err)
    }
    return user, nil
}

func (s *UserService) UpdateUser(ctx context.Context, id int64, input CreateUserInput) (*repository.User, error) {
    user, err := s.repo.FindByID(ctx, id)
    if err != nil {
        if errors.Is(err, repository.ErrNotFound) {
            return nil, ErrUserNotFound
        }
        return nil, fmt.Errorf("find user: %w", err)
    }
    
    user.Email = input.Email
    user.Name = input.Name
    
    if err := s.repo.Update(ctx, user); err != nil {
        return nil, fmt.Errorf("update user: %w", err)
    }
    
    return user, nil
}

func (s *UserService) DeleteUser(ctx context.Context, id int64) error {
    if err := s.repo.Delete(ctx, id); err != nil {
        if errors.Is(err, repository.ErrNotFound) {
            return ErrUserNotFound
        }
        return fmt.Errorf("delete user: %w", err)
    }
    return nil
}
```

---

## Middleware Stack

```go
// internal/middleware/middleware.go
package middleware

import (
    "context"
    "log/slog"
    "net/http"
    "runtime/debug"
    "time"
    
    "github.com/google/uuid"
)

type contextKey string

const RequestIDKey contextKey = "request_id"

// RequestID adds unique ID to each request
func RequestID(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        id := r.Header.Get("X-Request-ID")
        if id == "" {
            id = uuid.New().String()
        }
        
        ctx := context.WithValue(r.Context(), RequestIDKey, id)
        w.Header().Set("X-Request-ID", id)
        
        next.ServeHTTP(w, r.WithContext(ctx))
    })
}

// Logger logs request details
func Logger(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        
        // Wrap ResponseWriter to capture status
        ww := &responseWriter{ResponseWriter: w, statusCode: 200}
        
        next.ServeHTTP(ww, r)
        
        requestID, _ := r.Context().Value(RequestIDKey).(string)
        
        slog.Info("request completed",
            "request_id", requestID,
            "method", r.Method,
            "path", r.URL.Path,
            "status", ww.statusCode,
            "duration_ms", time.Since(start).Milliseconds(),
            "remote_addr", r.RemoteAddr,
        )
    })
}

// Recoverer catches panics and returns 500
func Recoverer(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if rec := recover(); rec != nil {
                requestID, _ := r.Context().Value(RequestIDKey).(string)
                
                slog.Error("panic recovered",
                    "request_id", requestID,
                    "panic", rec,
                    "stack", string(debug.Stack()),
                )
                
                http.Error(w, "Internal Server Error", http.StatusInternalServerError)
            }
        }()
        
        next.ServeHTTP(w, r)
    })
}

// Auth validates JWT token
func Auth(secret string) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            token := r.Header.Get("Authorization")
            if token == "" {
                http.Error(w, "Unauthorized", http.StatusUnauthorized)
                return
            }
            
            // Validate token (implement your logic)
            claims, err := validateToken(token, secret)
            if err != nil {
                http.Error(w, "Unauthorized", http.StatusUnauthorized)
                return
            }
            
            ctx := context.WithValue(r.Context(), "user_id", claims.UserID)
            next.ServeHTTP(w, r.WithContext(ctx))
        })
    }
}

type responseWriter struct {
    http.ResponseWriter
    statusCode int
}

func (w *responseWriter) WriteHeader(code int) {
    w.statusCode = code
    w.ResponseWriter.WriteHeader(code)
}
```

---

## Worker/Consumer

```go
// internal/worker/worker.go
package worker

import (
    "context"
    "log/slog"
    "sync"
    "time"
)

type Job struct {
    ID      string
    Payload []byte
}

type JobProcessor interface {
    Process(ctx context.Context, job Job) error
}

type Worker struct {
    jobs      chan Job
    processor JobProcessor
    workers   int
    wg        sync.WaitGroup
}

func NewWorker(processor JobProcessor, workers int, bufferSize int) *Worker {
    return &Worker{
        jobs:      make(chan Job, bufferSize),
        processor: processor,
        workers:   workers,
    }
}

func (w *Worker) Start(ctx context.Context) {
    for i := 0; i < w.workers; i++ {
        w.wg.Add(1)
        go w.worker(ctx, i)
    }
    slog.Info("workers started", "count", w.workers)
}

func (w *Worker) worker(ctx context.Context, id int) {
    defer w.wg.Done()
    
    for {
        select {
        case <-ctx.Done():
            slog.Info("worker stopping", "worker_id", id)
            return
            
        case job, ok := <-w.jobs:
            if !ok {
                slog.Info("job channel closed", "worker_id", id)
                return
            }
            
            w.processJob(ctx, id, job)
        }
    }
}

func (w *Worker) processJob(ctx context.Context, workerID int, job Job) {
    start := time.Now()
    
    // Create timeout context for individual job
    jobCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
    defer cancel()
    
    if err := w.processor.Process(jobCtx, job); err != nil {
        slog.Error("job failed",
            "worker_id", workerID,
            "job_id", job.ID,
            "error", err,
            "duration_ms", time.Since(start).Milliseconds(),
        )
        return
    }
    
    slog.Info("job completed",
        "worker_id", workerID,
        "job_id", job.ID,
        "duration_ms", time.Since(start).Milliseconds(),
    )
}

func (w *Worker) Submit(job Job) {
    w.jobs <- job
}

func (w *Worker) Stop() {
    close(w.jobs)
    w.wg.Wait()
    slog.Info("all workers stopped")
}
```

---

## Graceful Shutdown

```go
// pkg/server/server.go
package server

import (
    "context"
    "log/slog"
    "net/http"
    "os"
    "os/signal"
    "syscall"
    "time"
)

type Server struct {
    httpServer *http.Server
    cleanup    []func() error
}

func New(addr string, handler http.Handler) *Server {
    return &Server{
        httpServer: &http.Server{
            Addr:         addr,
            Handler:      handler,
            ReadTimeout:  15 * time.Second,
            WriteTimeout: 15 * time.Second,
            IdleTimeout:  60 * time.Second,
        },
    }
}

func (s *Server) AddCleanup(fn func() error) {
    s.cleanup = append(s.cleanup, fn)
}

func (s *Server) Run() error {
    // Error channel for server errors
    errCh := make(chan error, 1)
    
    // Start server
    go func() {
        slog.Info("starting server", "addr", s.httpServer.Addr)
        if err := s.httpServer.ListenAndServe(); err != http.ErrServerClosed {
            errCh <- err
        }
    }()
    
    // Wait for interrupt or error
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
    
    select {
    case err := <-errCh:
        return err
    case sig := <-quit:
        slog.Info("received signal", "signal", sig)
    }
    
    // Graceful shutdown
    slog.Info("shutting down server...")
    
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    
    // Shutdown HTTP server
    if err := s.httpServer.Shutdown(ctx); err != nil {
        slog.Error("server shutdown error", "error", err)
    }
    
    // Run cleanup functions
    for _, fn := range s.cleanup {
        if err := fn(); err != nil {
            slog.Error("cleanup error", "error", err)
        }
    }
    
    slog.Info("server stopped gracefully")
    return nil
}
```

---

## Config Management

```go
// internal/config/config.go
package config

import (
    "fmt"
    "time"
    
    "github.com/caarlos0/env/v10"
)

type Config struct {
    // Server
    ServerAddr    string        `env:"SERVER_ADDR" envDefault:":8080"`
    ReadTimeout   time.Duration `env:"READ_TIMEOUT" envDefault:"15s"`
    WriteTimeout  time.Duration `env:"WRITE_TIMEOUT" envDefault:"15s"`
    
    // Database
    DatabaseURL     string        `env:"DATABASE_URL,required"`
    DatabaseMaxConn int           `env:"DATABASE_MAX_CONN" envDefault:"25"`
    DatabaseMaxIdle int           `env:"DATABASE_MAX_IDLE" envDefault:"5"`
    
    // Redis
    RedisURL string `env:"REDIS_URL" envDefault:"localhost:6379"`
    
    // Auth
    JWTSecret     string        `env:"JWT_SECRET,required"`
    JWTExpiration time.Duration `env:"JWT_EXPIRATION" envDefault:"24h"`
    
    // Feature Flags
    EnableMetrics bool `env:"ENABLE_METRICS" envDefault:"true"`
    EnablePprof   bool `env:"ENABLE_PPROF" envDefault:"false"`
    
    // Environment
    Environment string `env:"ENVIRONMENT" envDefault:"development"`
    LogLevel    string `env:"LOG_LEVEL" envDefault:"info"`
}

func Load() (*Config, error) {
    cfg := &Config{}
    if err := env.Parse(cfg); err != nil {
        return nil, fmt.Errorf("parse config: %w", err)
    }
    return cfg, nil
}

func (c *Config) IsDevelopment() bool {
    return c.Environment == "development"
}

func (c *Config) IsProduction() bool {
    return c.Environment == "production"
}
```

---

## Custom Error Types

```go
// pkg/apperror/error.go
package apperror

import (
    "fmt"
    "net/http"
)

type AppError struct {
    Code    string `json:"code"`
    Message string `json:"message"`
    Status  int    `json:"-"`
    Err     error  `json:"-"`
}

func (e *AppError) Error() string {
    if e.Err != nil {
        return fmt.Sprintf("%s: %v", e.Message, e.Err)
    }
    return e.Message
}

func (e *AppError) Unwrap() error {
    return e.Err
}

// Constructors
func NotFound(resource string) *AppError {
    return &AppError{
        Code:    "NOT_FOUND",
        Message: fmt.Sprintf("%s not found", resource),
        Status:  http.StatusNotFound,
    }
}

func BadRequest(message string) *AppError {
    return &AppError{
        Code:    "BAD_REQUEST",
        Message: message,
        Status:  http.StatusBadRequest,
    }
}

func Unauthorized(message string) *AppError {
    return &AppError{
        Code:    "UNAUTHORIZED",
        Message: message,
        Status:  http.StatusUnauthorized,
    }
}

func Internal(err error) *AppError {
    return &AppError{
        Code:    "INTERNAL_ERROR",
        Message: "An internal error occurred",
        Status:  http.StatusInternalServerError,
        Err:     err,
    }
}

func Conflict(message string) *AppError {
    return &AppError{
        Code:    "CONFLICT",
        Message: message,
        Status:  http.StatusConflict,
    }
}

// Handler helper
func WriteError(w http.ResponseWriter, err *AppError) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(err.Status)
    json.NewEncoder(w).Encode(err)
}
```
