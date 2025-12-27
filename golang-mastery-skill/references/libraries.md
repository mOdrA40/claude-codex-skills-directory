# Battle-Tested Go Libraries

## Table of Contents
1. [HTTP & Web](#http--web)
2. [Database](#database)
3. [Configuration](#configuration)
4. [Logging](#logging)
5. [Testing](#testing)
6. [Validation](#validation)
7. [Authentication & Security](#authentication--security)
8. [Message Queue & Events](#message-queue--events)
9. [Utilities](#utilities)
10. [Observability](#observability)

## HTTP & Web

### Chi - Lightweight Router (RECOMMENDED)
```go
import "github.com/go-chi/chi/v5"

r := chi.NewRouter()
r.Use(middleware.Logger)
r.Use(middleware.Recoverer)
r.Use(middleware.Timeout(60 * time.Second))

r.Route("/api/v1", func(r chi.Router) {
    r.Get("/users", listUsers)
    r.Post("/users", createUser)
    r.Route("/users/{id}", func(r chi.Router) {
        r.Get("/", getUser)
        r.Put("/", updateUser)
        r.Delete("/", deleteUser)
    })
})
```
**Why Chi:** Go-idiomatic, minimal, context-based, excellent middleware ecosystem.

### Gin - Feature-Rich Framework
```go
import "github.com/gin-gonic/gin"

r := gin.Default()
r.GET("/users/:id", func(c *gin.Context) {
    id := c.Param("id")
    c.JSON(200, gin.H{"id": id})
})
```
**Why Gin:** Fastest, built-in validation, great for rapid development.

### Fiber - Express-Like (High Performance)
```go
import "github.com/gofiber/fiber/v2"

app := fiber.New()
app.Get("/users/:id", func(c *fiber.Ctx) error {
    return c.JSON(fiber.Map{"id": c.Params("id")})
})
```
**Why Fiber:** Fastest benchmarks, Express.js familiarity.

## Database

### sqlx - Extended SQL (RECOMMENDED)
```go
import "github.com/jmoiron/sqlx"

type User struct {
    ID    int    `db:"id"`
    Name  string `db:"name"`
    Email string `db:"email"`
}

var users []User
err := db.Select(&users, "SELECT * FROM users WHERE active = $1", true)

_, err = db.NamedExec(`INSERT INTO users (name, email) VALUES (:name, :email)`, user)
```
**Why sqlx:** SQL power + type safety, no ORM overhead.

### pgx - PostgreSQL Driver (RECOMMENDED)
```go
import "github.com/jackc/pgx/v5/pgxpool"

pool, err := pgxpool.New(ctx, "postgres://user:pass@localhost:5432/db")

batch := &pgx.Batch{}
batch.Queue("INSERT INTO users(name) VALUES($1)", "user1")
batch.Queue("INSERT INTO users(name) VALUES($1)", "user2")
results := pool.SendBatch(ctx, batch)
```
**Why pgx:** Native PostgreSQL, connection pooling, COPY support.

### GORM - Full ORM
```go
import "gorm.io/gorm"

type User struct {
    gorm.Model
    Name  string
    Email string `gorm:"uniqueIndex"`
}

db.AutoMigrate(&User{})
db.Where("name LIKE ?", "%john%").Find(&users)
```
**When GORM:** Rapid prototyping, complex relations.
**Avoid GORM:** High-performance systems, complex queries.

### go-migrate - Database Migrations
```bash
go install -tags 'postgres' github.com/golang-migrate/migrate/v4/cmd/migrate@latest
migrate create -ext sql -dir migrations -seq add_users_table
migrate -path migrations -database "postgres://..." up
```

## Configuration

### envconfig - Simple Env Vars (RECOMMENDED for Microservices)
```go
import "github.com/kelseyhightower/envconfig"

type Config struct {
    Port        int           `envconfig:"PORT" default:"8080"`
    DatabaseURL string        `envconfig:"DATABASE_URL" required:"true"`
    Timeout     time.Duration `envconfig:"TIMEOUT" default:"30s"`
}

var cfg Config
envconfig.Process("", &cfg)
```
**Why envconfig:** 12-factor app compliant, simple, Docker-friendly.

### Viper - Feature-Rich (Complex Apps)
```go
import "github.com/spf13/viper"

viper.SetConfigName("config")
viper.AutomaticEnv()
viper.SetDefault("server.port", 8080)
viper.ReadInConfig()
```

## Logging

### zerolog - Fastest Structured Logger (RECOMMENDED)
```go
import "github.com/rs/zerolog"

logger := zerolog.New(os.Stdout).With().Timestamp().Str("service", "api").Logger()

logger.Info().
    Str("user_id", "123").
    Int("status", 200).
    Dur("latency", time.Since(start)).
    Msg("request completed")
```

### slog - Standard Library (Go 1.21+)
```go
import "log/slog"

logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
logger.Info("request completed", slog.String("user_id", "123"))
```
**Why slog:** No dependencies, built-in.

## Testing

### testify - Assertions & Mocks (RECOMMENDED)
```go
import (
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    "github.com/stretchr/testify/mock"
)

func TestCalculate(t *testing.T) {
    assert.Equal(t, 42, Calculate(6, 7))
    
    result, err := SomeFunction()
    require.NoError(t, err)
    require.NotNil(t, result)
}
```

### testcontainers-go - Integration Tests
```go
import "github.com/testcontainers/testcontainers-go"

container, _ := testcontainers.GenericContainer(ctx, testcontainers.GenericContainerRequest{
    ContainerRequest: testcontainers.ContainerRequest{
        Image: "postgres:16-alpine",
        Env:   map[string]string{"POSTGRES_PASSWORD": "test"},
    },
    Started: true,
})
defer container.Terminate(ctx)
```

## Validation

### validator/v10 (RECOMMENDED)
```go
import "github.com/go-playground/validator/v10"

type CreateUserRequest struct {
    Name     string `validate:"required,min=2,max=50"`
    Email    string `validate:"required,email"`
    Password string `validate:"required,min=8"`
}

validate := validator.New()
err := validate.Struct(req)
```

## Authentication & Security

### golang-jwt/jwt - JWT Handling
```go
import "github.com/golang-jwt/jwt/v5"

claims := jwt.MapClaims{"user_id": "123", "exp": time.Now().Add(24 * time.Hour).Unix()}
token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
tokenString, _ := token.SignedString([]byte(secretKey))
```

### bcrypt - Password Hashing
```go
import "golang.org/x/crypto/bcrypt"

hash, _ := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
err := bcrypt.CompareHashAndPassword(hash, []byte(password))
```

## Message Queue & Events

### asynq - Task Queue (Redis-based)
```go
import "github.com/hibiken/asynq"

client := asynq.NewClient(asynq.RedisClientOpt{Addr: "localhost:6379"})
task := asynq.NewTask("email:send", payload)
client.Enqueue(task, asynq.ProcessIn(10*time.Second))
```

### watermill - Event-Driven
```go
import "github.com/ThreeDotsLabs/watermill"

publisher.Publish("user.created", message.NewMessage(uuid.NewString(), payload))
```

## Utilities

### lo - Lodash for Go (Generics)
```go
import "github.com/samber/lo"

evens := lo.Filter([]int{1, 2, 3, 4}, func(n int, _ int) bool { return n%2 == 0 })
doubled := lo.Map([]int{1, 2, 3}, func(n int, _ int) int { return n * 2 })
```

### retry-go - Retry Logic
```go
import "github.com/avast/retry-go/v4"

err := retry.Do(func() error { return fetchData() },
    retry.Attempts(3), retry.Delay(time.Second), retry.DelayType(retry.BackOffDelay))
```

### uuid - UUID Generation
```go
import "github.com/google/uuid"

id := uuid.NewString()
```

## Observability

### OpenTelemetry - Tracing & Metrics
```go
import "go.opentelemetry.io/otel"

tracer := otel.Tracer("myapp")
ctx, span := tracer.Start(ctx, "operation-name")
defer span.End()
```

### prometheus/client_golang - Metrics
```go
import "github.com/prometheus/client_golang/prometheus"

httpRequestsTotal.WithLabelValues("GET", "/users", "200").Inc()
```

## Essential Stack Installation

```bash
go get github.com/go-chi/chi/v5
go get github.com/jmoiron/sqlx
go get github.com/jackc/pgx/v5
go get github.com/kelseyhightower/envconfig
go get github.com/rs/zerolog
go get github.com/go-playground/validator/v10
go get github.com/golang-jwt/jwt/v5
go get github.com/stretchr/testify
go get github.com/samber/lo
go get github.com/google/uuid
```
