# Docker Best Practices untuk Go

## Table of Contents
1. [Production Dockerfile](#production-dockerfile)
2. [Development Dockerfile](#development-dockerfile)
3. [Docker Compose](#docker-compose)
4. [.dockerignore](#dockerignore)
5. [Security Hardening](#security-hardening)
6. [Multi-Architecture Builds](#multi-architecture-builds)
7. [Common Mistakes](#common-mistakes)

## Production Dockerfile

```dockerfile
# =============================================================================
# PRODUCTION DOCKERFILE - Multi-stage build
# Final image size: ~10-20MB (vs ~800MB tanpa multi-stage)
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build
# -----------------------------------------------------------------------------
FROM golang:1.22-alpine AS builder

# Install dependencies untuk CGO jika diperlukan
RUN apk add --no-cache git ca-certificates tzdata

WORKDIR /app

# Cache dependencies terlebih dahulu
COPY go.mod go.sum ./
RUN go mod download && go mod verify

# Copy source code
COPY . .

# Build dengan optimizations
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-w -s -X main.Version=${VERSION:-dev}" \
    -o /app/server \
    ./cmd/api

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM alpine:3.19

# Security: non-root user
RUN addgroup -g 1001 appgroup && \
    adduser -D -u 1001 -G appgroup appuser

# Install CA certificates untuk HTTPS calls
RUN apk add --no-cache ca-certificates tzdata

WORKDIR /app

# Copy binary dari builder
COPY --from=builder /app/server .

# Copy config jika ada
COPY --from=builder /app/config ./config

# Set ownership
RUN chown -R appuser:appgroup /app

# Switch ke non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# Run
ENTRYPOINT ["./server"]
```

## Development Dockerfile

```dockerfile
# =============================================================================
# DEVELOPMENT DOCKERFILE - dengan hot reload
# =============================================================================

FROM golang:1.22-alpine

# Install tools untuk development
RUN go install github.com/air-verse/air@latest && \
    go install github.com/go-delve/delve/cmd/dlv@latest

WORKDIR /app

# Copy go.mod dulu untuk caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source (akan di-mount saat development)
COPY . .

# Expose app port dan debugger port
EXPOSE 8080 2345

# Hot reload dengan Air
CMD ["air", "-c", ".air.toml"]
```

### .air.toml (Hot Reload Config)

```toml
root = "."
tmp_dir = "tmp"

[build]
  cmd = "go build -o ./tmp/main ./cmd/api"
  bin = "./tmp/main"
  delay = 1000
  exclude_dir = ["assets", "tmp", "vendor", "testdata"]
  exclude_file = []
  exclude_regex = ["_test.go"]
  exclude_unchanged = false
  follow_symlink = false
  full_bin = ""
  include_dir = []
  include_ext = ["go", "tpl", "tmpl", "html"]
  kill_delay = "2s"
  log = "build-errors.log"
  send_interrupt = false
  stop_on_error = true

[log]
  time = false

[color]
  build = "yellow"
  main = "magenta"
  runner = "green"
  watcher = "cyan"

[misc]
  clean_on_exit = true
```

## Docker Compose

### Production

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile
    image: myapp:${VERSION:-latest}
    container_name: myapp-api
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      - APP_ENV=production
      - DATABASE_URL=postgres://user:pass@db:5432/mydb?sslmode=disable
      - REDIS_URL=redis://redis:6379
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - backend
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M

  db:
    image: postgres:16-alpine
    container_name: myapp-db
    restart: unless-stopped
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: mydb
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./migrations/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    networks:
      - backend
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d mydb"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: myapp-redis
    restart: unless-stopped
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    networks:
      - backend
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
  redis_data:

networks:
  backend:
    driver: bridge
```

### Development

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.dev
    volumes:
      - .:/app                          # Mount source code
      - go-modules:/go/pkg/mod          # Cache modules
    ports:
      - "8080:8080"
      - "2345:2345"                      # Debugger port
    environment:
      - APP_ENV=development
      - DATABASE_URL=postgres://user:pass@db:5432/mydb?sslmode=disable
    depends_on:
      - db
      - redis

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: mydb
    ports:
      - "5432:5432"                      # Expose untuk local tools

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  go-modules:
```

## .dockerignore

```
# Git
.git
.gitignore

# IDE
.idea
.vscode
*.swp
*.swo

# Build artifacts
bin/
dist/
tmp/

# Test artifacts
coverage.out
*.test

# Documentation
*.md
!README.md
docs/

# Development files
.env.local
.env.development
docker-compose.dev.yml
Dockerfile.dev

# CI/CD
.github/
.gitlab-ci.yml
Jenkinsfile

# Misc
Makefile
*.log
```

## Security Hardening

### 1. Non-Root User (WAJIB)

```dockerfile
# NEVER run as root in production
RUN addgroup -g 1001 appgroup && \
    adduser -D -u 1001 -G appgroup appuser
USER appuser
```

### 2. Read-Only Filesystem

```yaml
# docker-compose.yml
services:
  api:
    read_only: true
    tmpfs:
      - /tmp:size=64M
    volumes:
      - ./logs:/app/logs  # Hanya folder yang perlu write
```

### 3. Security Options

```yaml
services:
  api:
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE  # Hanya jika perlu bind port < 1024
```

### 4. Scan Vulnerabilities

```bash
# Scan image untuk vulnerabilities
docker scout cves myapp:latest

# Atau gunakan trivy
trivy image myapp:latest
```

## Multi-Architecture Builds

```bash
# Setup buildx
docker buildx create --name mybuilder --use

# Build untuk multiple architectures
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -t myapp:latest \
    --push \
    .
```

### Dockerfile untuk Multi-Arch

```dockerfile
FROM --platform=$BUILDPLATFORM golang:1.22-alpine AS builder

ARG TARGETPLATFORM
ARG BUILDPLATFORM
ARG TARGETOS
ARG TARGETARCH

WORKDIR /app
COPY . .

RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build \
    -o /app/server ./cmd/api
```

## Common Mistakes

### ❌ Mistake 1: Using `latest` Tag

```dockerfile
# ❌ BAD - unpredictable
FROM golang:latest

# ✅ GOOD - reproducible
FROM golang:1.22-alpine
```

### ❌ Mistake 2: Not Using Multi-Stage

```dockerfile
# ❌ BAD - 800MB+ image dengan build tools
FROM golang:1.22
COPY . .
RUN go build -o /app ./cmd/api
CMD ["/app"]

# ✅ GOOD - 10-20MB image
# (lihat Production Dockerfile di atas)
```

### ❌ Mistake 3: Copying Before Downloading Dependencies

```dockerfile
# ❌ BAD - setiap code change re-download semua deps
COPY . .
RUN go mod download

# ✅ GOOD - deps cached jika go.mod tidak berubah
COPY go.mod go.sum ./
RUN go mod download
COPY . .
```

### ❌ Mistake 4: Running as Root

```dockerfile
# ❌ BAD - security risk
CMD ["./server"]

# ✅ GOOD - run as non-root
USER appuser
CMD ["./server"]
```

### ❌ Mistake 5: No Health Check

```yaml
# ❌ BAD - orchestrator tidak tau app status
services:
  api:
    image: myapp

# ✅ GOOD - proper health monitoring
services:
  api:
    image: myapp
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Useful Commands

```bash
# Build
docker build -t myapp:v1 -f docker/Dockerfile .

# Run dengan resource limits
docker run -d --name myapp \
    --cpus=2 --memory=512m \
    -p 8080:8080 \
    myapp:v1

# Check logs
docker logs -f myapp

# Shell into container (debugging)
docker exec -it myapp sh

# Check resource usage
docker stats myapp

# Prune unused resources
docker system prune -af
```
