#!/usr/bin/env python3
"""
Initialize a production-grade Go project with best practices.
Usage: python init_project.py <project-name> [--path <output-directory>]
"""

import os
import sys
import argparse

MAKEFILE = '''# Project variables
PROJECT_NAME := {project_name}
BINARY_NAME := $(PROJECT_NAME)
DOCKER_IMAGE := $(PROJECT_NAME)
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")

# Go variables
GOFLAGS := -ldflags="-w -s -X main.Version=$(VERSION)"

.PHONY: all build run test lint clean docker-build docker-run help

all: lint test build

## Build
build:
	@echo "Building..."
	CGO_ENABLED=0 go build $(GOFLAGS) -o bin/$(BINARY_NAME) ./cmd/api

run:
	go run ./cmd/api

## Testing
test:
	go test -v -race -coverprofile=coverage.out ./...

test-short:
	go test -v -short ./...

coverage:
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

## Linting
lint:
	golangci-lint run ./...

fmt:
	gofmt -s -w .
	goimports -w .

## Database
migrate-up:
	migrate -path migrations -database "$(DATABASE_URL)" up

migrate-down:
	migrate -path migrations -database "$(DATABASE_URL)" down 1

migrate-create:
	@read -p "Migration name: " name; \\
	migrate create -ext sql -dir migrations -seq $$name

## Docker
docker-build:
	docker build -t $(DOCKER_IMAGE):$(VERSION) -f docker/Dockerfile .

docker-run:
	docker-compose -f docker/docker-compose.yml up -d

docker-stop:
	docker-compose -f docker/docker-compose.yml down

## Clean
clean:
	rm -rf bin/ tmp/ coverage.out coverage.html

## Help
help:
	@echo "Available targets:"
	@echo "  build        - Build the binary"
	@echo "  run          - Run the application"
	@echo "  test         - Run tests with race detector"
	@echo "  lint         - Run linters"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run with Docker Compose"
'''

GOLANGCI_LINT = '''# .golangci.yml
run:
  timeout: 5m
  tests: true

linters:
  enable:
    - errcheck
    - gosimple
    - govet
    - ineffassign
    - staticcheck
    - unused
    - bodyclose
    - contextcheck
    - durationcheck
    - errname
    - errorlint
    - exhaustive
    - exportloopref
    - gofmt
    - goimports
    - goconst
    - gocritic
    - gosec
    - misspell
    - nilerr
    - nilnil
    - noctx
    - prealloc
    - revive
    - sqlclosecheck
    - unconvert
    - unparam
    - wastedassign

linters-settings:
  govet:
    check-shadowing: true
  goconst:
    min-len: 3
    min-occurrences: 3
  gosec:
    severity: medium
    confidence: medium
  revive:
    rules:
      - name: blank-imports
      - name: context-as-argument
      - name: context-keys-type
      - name: error-return
      - name: error-strings
      - name: error-naming
      - name: exported
      - name: increment-decrement
      - name: var-naming
      - name: package-comments

issues:
  exclude-rules:
    - path: _test\\.go
      linters:
        - gosec
        - goconst
'''

DOCKERFILE = '''# =============================================================================
# Production Dockerfile - Multi-stage build
# =============================================================================

FROM golang:1.22-alpine AS builder

RUN apk add --no-cache git ca-certificates tzdata

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download && go mod verify

COPY . .

ARG VERSION=dev
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \\
    -ldflags="-w -s -X main.Version=${VERSION}" \\
    -o /app/server \\
    ./cmd/api

# -----------------------------------------------------------------------------
FROM alpine:3.19

RUN addgroup -g 1001 appgroup && \\
    adduser -D -u 1001 -G appgroup appuser

RUN apk add --no-cache ca-certificates tzdata

WORKDIR /app

COPY --from=builder /app/server .
COPY --from=builder /app/config ./config

RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

ENTRYPOINT ["./server"]
'''

DOCKER_COMPOSE = '''version: '3.8'

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    image: {project_name}:${{VERSION:-latest}}
    container_name: {project_name}-api
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      - APP_ENV=production
      - DATABASE_URL=postgres://user:pass@db:5432/{project_name}?sslmode=disable
    depends_on:
      db:
        condition: service_healthy
    networks:
      - backend
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:16-alpine
    container_name: {project_name}-db
    restart: unless-stopped
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: {project_name}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - backend
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d {project_name}"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:

networks:
  backend:
    driver: bridge
'''

MAIN_GO = '''package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
)

var Version = "dev"

func main() {
	r := chi.NewRouter()

	// Middleware
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)
	r.Use(middleware.Timeout(60 * time.Second))

	// Routes
	r.Get("/health", healthHandler)
	r.Get("/version", versionHandler)

	// API routes
	r.Route("/api/v1", func(r chi.Router) {
		// Add your routes here
	})

	// Server
	server := &http.Server{
		Addr:         ":8080",
		Handler:      r,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Graceful shutdown
	go func() {
		log.Printf("Server starting on %s (version: %s)", server.Addr, Version)
		if err := server.ListenAndServe(); err != http.ErrServerClosed {
			log.Fatalf("Server error: %v", err)
		}
	}()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	log.Println("Shutdown signal received")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		log.Printf("Server shutdown error: %v", err)
	}

	log.Println("Server stopped")
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(`{"status":"ok"}`))
}

func versionHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(`{"version":"` + Version + `"}`))
}
'''

GO_MOD = '''module {module_name}

go 1.22

require (
	github.com/go-chi/chi/v5 v5.0.12
)
'''

GITIGNORE = '''# Binaries
bin/
*.exe
*.dll
*.so
*.dylib

# Test
*.test
coverage.out
coverage.html

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Environment
.env
.env.local
.env.*.local

# Build
tmp/
dist/

# Logs
*.log
'''

DOCKERIGNORE = '''.git
.gitignore
.idea
.vscode
bin/
dist/
tmp/
coverage.out
*.md
!README.md
docker-compose*.yml
Makefile
.env*
'''

ENV_EXAMPLE = '''APP_ENV=development
PORT=8080
DATABASE_URL=postgres://user:pass@localhost:5432/dbname?sslmode=disable
LOG_LEVEL=debug
'''


def create_project(project_name: str, base_path: str):
    """Create a new Go project with production-grade structure."""
    
    project_path = os.path.join(base_path, project_name)
    
    if os.path.exists(project_path):
        print(f"Error: Directory already exists: {project_path}")
        sys.exit(1)
    
    # Create directory structure
    dirs = [
        "cmd/api",
        "internal/domain",
        "internal/usecase",
        "internal/repository/postgres",
        "internal/handler/http",
        "internal/pkg/validator",
        "internal/pkg/logger",
        "pkg",
        "config",
        "migrations",
        "scripts",
        "docker",
    ]
    
    for d in dirs:
        os.makedirs(os.path.join(project_path, d), exist_ok=True)
        print(f"Created: {d}/")
    
    # Create files
    files = {
        "Makefile": MAKEFILE.format(project_name=project_name),
        ".golangci.yml": GOLANGCI_LINT,
        "docker/Dockerfile": DOCKERFILE,
        "docker/docker-compose.yml": DOCKER_COMPOSE.format(project_name=project_name),
        "cmd/api/main.go": MAIN_GO,
        "go.mod": GO_MOD.format(module_name=f"github.com/yourorg/{project_name}"),
        ".gitignore": GITIGNORE,
        ".dockerignore": DOCKERIGNORE,
        ".env.example": ENV_EXAMPLE,
    }
    
    for filepath, content in files.items():
        full_path = os.path.join(project_path, filepath)
        with open(full_path, 'w') as f:
            f.write(content)
        print(f"Created: {filepath}")
    
    # Create placeholder files
    placeholders = [
        ("internal/domain/.gitkeep", ""),
        ("internal/usecase/.gitkeep", ""),
        ("internal/repository/postgres/.gitkeep", ""),
        ("internal/handler/http/.gitkeep", ""),
        ("pkg/.gitkeep", ""),
        ("migrations/.gitkeep", ""),
    ]
    
    for filepath, content in placeholders:
        full_path = os.path.join(project_path, filepath)
        with open(full_path, 'w') as f:
            f.write(content)
    
    print(f"\n✅ Project '{project_name}' created at {project_path}")
    print("\nNext steps:")
    print(f"  cd {project_path}")
    print("  go mod tidy")
    print("  make run")


def main():
    parser = argparse.ArgumentParser(description="Initialize a production-grade Go project")
    parser.add_argument("project_name", help="Name of the project")
    parser.add_argument("--path", default=".", help="Output directory (default: current directory)")
    
    args = parser.parse_args()
    
    create_project(args.project_name, args.path)


if __name__ == "__main__":
    main()
