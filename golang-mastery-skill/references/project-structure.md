# Project Structure

Struktur folder yang battle-tested untuk berbagai skala project.

## Small Service (< 2000 LOC)

```
myservice/
├── main.go              # Entry point, wiring
├── handler.go           # HTTP handlers
├── service.go           # Business logic
├── repository.go        # Data access
├── model.go             # Domain types
├── config.go            # Configuration
├── Dockerfile
├── docker-compose.yml
└── go.mod
```

## Medium Service (2000-10000 LOC)

```
myservice/
├── cmd/
│   └── api/
│       └── main.go           # Entry point
├── internal/                  # Private packages
│   ├── config/
│   ├── handler/               # HTTP layer
│   ├── service/               # Business logic
│   ├── repository/            # Data access
│   ├── model/                 # Domain entities
│   └── pkg/                   # Shared utilities
├── pkg/                       # Public packages
├── migrations/
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── go.mod
```

## Large/Monorepo (10000+ LOC)

```
platform/
├── cmd/                       # Semua executables
│   ├── api-gateway/
│   ├── user-service/
│   └── worker/
├── internal/                  # Private shared code
│   ├── platform/              # Infrastructure (db, cache, queue)
│   ├── user/                  # User domain
│   └── order/                 # Order domain
├── pkg/                       # Public shared packages
├── api/                       # API definitions (openapi, proto)
├── deployments/               # Docker, k8s, terraform
├── Makefile
└── go.mod
```

## File Naming

```go
// ✅ BENAR
user_service.go      // Snake case
user_service_test.go // Test file

// ❌ SALAH
userService.go       // Camel case
user-service.go      // Kebab case
```

## Package Guidelines

```go
// ✅ Singular, lowercase
package user
package order

// ❌ SALAH
package users     // Plural
package userService // Camel case
```

## Dependency Direction

```
handler → service → repository → model
    ↓         ↓          ↓
  model     model      model
```

Tidak boleh: child → parent (circular dependency)
