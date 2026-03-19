# Docker Patterns untuk Bun.js Production

## Table of Contents
1. [Multi-stage Build](#multi-stage-build)
2. [Development Setup](#development-setup)
3. [Production Optimization](#production-optimization)
4. [Security Hardening](#security-hardening)
5. [Docker Compose Patterns](#docker-compose)
6. [CI/CD Integration](#cicd)

---

## Multi-stage Build

### Basic Production Dockerfile
```dockerfile
# ========================================
# Stage 1: Dependencies
# ========================================
FROM oven/bun:1-alpine AS deps
WORKDIR /app

# Copy hanya package files untuk cache layer
COPY package.json bun.lockb ./
RUN bun install --frozen-lockfile

# ========================================
# Stage 2: Builder
# ========================================
FROM oven/bun:1-alpine AS builder
WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build aplikasi
RUN bun run build

# Prune dev dependencies
RUN bun install --production --frozen-lockfile

# ========================================
# Stage 3: Runner (Production)
# ========================================
FROM oven/bun:1-alpine AS runner
WORKDIR /app

# Security: Non-root user
RUN addgroup -g 1001 -S appgroup && \
    adduser -S appuser -u 1001 -G appgroup

# Set environment
ENV NODE_ENV=production
ENV TZ=Asia/Jakarta

# Copy built files
COPY --from=builder --chown=appuser:appgroup /app/dist ./dist
COPY --from=builder --chown=appuser:appgroup /app/node_modules ./node_modules
COPY --from=builder --chown=appuser:appgroup /app/package.json ./

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

# Start application
CMD ["bun", "run", "dist/index.js"]
```

### Ultra-minimal dengan Distroless
```dockerfile
# Build stage
FROM oven/bun:1-alpine AS builder
WORKDIR /app
COPY package.json bun.lockb ./
RUN bun install --frozen-lockfile --production=false
COPY . .
RUN bun build src/index.ts --compile --outfile=server

# Production - distroless-like
FROM gcr.io/distroless/cc-debian12
WORKDIR /app
COPY --from=builder /app/server ./server
EXPOSE 3000
CMD ["./server"]
```

---

## Development Setup

### Development Dockerfile
```dockerfile
# Dockerfile.dev
FROM oven/bun:1-alpine

WORKDIR /app

# Install dev tools
RUN apk add --no-cache curl wget

# Copy package files
COPY package.json bun.lockb ./
RUN bun install

# Copy source (mounted volume akan override)
COPY . .

# Development port
EXPOSE 3000

# Hot reload dengan watch
CMD ["bun", "--watch", "src/index.ts"]
```

### docker-compose.dev.yml
```yaml
version: "3.9"

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "3000:3000"
      - "9229:9229"  # Debugger
    volumes:
      # Mount source untuk hot reload
      - .:/app
      # Named volume untuk node_modules (avoid conflict)
      - node_modules:/app/node_modules
    environment:
      - NODE_ENV=development
      - DATABASE_URL=postgres://postgres:postgres@db:5432/app_dev
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=debug
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - backend

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: app_dev
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - backend

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - backend

  # Optional: Database admin
  adminer:
    image: adminer
    ports:
      - "8080:8080"
    depends_on:
      - db
    networks:
      - backend

volumes:
  node_modules:
  postgres_data:
  redis_data:

networks:
  backend:
    driver: bridge
```

---

## Production Optimization

### .dockerignore (WAJIB!)
```
# Dependencies
node_modules
.pnp
.pnp.js

# Build outputs
dist
build
*.tsbuildinfo

# Testing
coverage
.nyc_output

# Development
.env.local
.env.development
.env*.local
docker-compose*.yml
Dockerfile.dev

# Logs
*.log
npm-debug.log*

# IDE
.idea
.vscode
*.swp
*.swo

# Git
.git
.gitignore

# Documentation
README.md
CHANGELOG.md
docs

# Tests
tests
__tests__
*.test.ts
*.spec.ts
vitest.config.ts
jest.config.ts

# Misc
.DS_Store
Thumbs.db
```

### Layer Caching Strategy
```dockerfile
# ❌ Bad - setiap perubahan kode rebuild semua
COPY . .
RUN bun install

# ✅ Good - maksimalkan cache layers
# Layer 1: Package files (jarang berubah)
COPY package.json bun.lockb ./
RUN bun install --frozen-lockfile

# Layer 2: Source code (sering berubah)
COPY src ./src
COPY tsconfig.json ./

# Layer 3: Build
RUN bun run build
```

### Image Size Reduction
```bash
# Check image size
docker images my-app

# Analyze layers
docker history my-app

# Dive tool untuk analisis detail
docker run --rm -it \
  -v /var/run/docker.sock:/var/run/docker.sock \
  wagoodman/dive:latest my-app
```

Ukuran ideal:
- Development image: < 500MB
- Production image: < 150MB
- Ultra-minimal: < 50MB

---

## Security Hardening

### Non-root User (WAJIB)
```dockerfile
# Create user dengan UID/GID specific
RUN addgroup -g 1001 -S appgroup && \
    adduser -S appuser -u 1001 -G appgroup

# Set ownership
COPY --chown=appuser:appgroup . .

# Switch user SEBELUM CMD
USER appuser
```

### Read-only Filesystem
```yaml
# docker-compose.prod.yml
services:
  api:
    read_only: true
    tmpfs:
      - /tmp
    volumes:
      # Only mount what's needed as read-only
      - ./config:/app/config:ro
```

### Resource Limits
```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 1G
        reservations:
          cpus: "0.5"
          memory: 256M
```

### Security Scanning
```bash
# Scan dengan Trivy
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image my-app:latest

# Scan dengan Snyk
docker scan my-app:latest
```

---

## Docker Compose

### Production Compose
```yaml
# docker-compose.prod.yml
version: "3.9"

services:
  api:
    image: ${REGISTRY}/my-app:${VERSION:-latest}
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: "1"
          memory: 512M
      update_config:
        parallelism: 1
        delay: 10s
        order: start-first
      rollback_config:
        parallelism: 0
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    networks:
      - backend
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:16-alpine
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: ${DB_NAME}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          memory: 1G
    networks:
      - backend

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - backend

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/certs:/etc/nginx/certs:ro
    depends_on:
      - api
    networks:
      - backend

volumes:
  postgres_data:
  redis_data:

networks:
  backend:
    driver: bridge
```

### Nginx Config untuk Reverse Proxy
```nginx
# nginx/nginx.conf
upstream api {
    least_conn;
    server api:3000;
}

server {
    listen 80;
    server_name _;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name _;

    ssl_certificate /etc/nginx/certs/cert.pem;
    ssl_certificate_key /etc/nginx/certs/key.pem;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    location / {
        proxy_pass http://api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /health {
        access_log off;
        proxy_pass http://api/health;
    }
}
```

---

## CI/CD

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Build and Deploy

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: oven-sh/setup-bun@v1
        with:
          bun-version: latest
      
      - run: bun install
      - run: bun run lint
      - run: bun run test

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4
      
      - uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - uses: docker/metadata-action@v5
        id: meta
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha,prefix=
            type=raw,value=latest,enable={{is_default_branch}}
      
      - uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - name: Deploy to production
        run: |
          # SSH dan deploy ke server
          # Atau trigger webhook ke platform
          echo "Deploying..."
```

### Makefile untuk Local Development
```makefile
# Makefile
.PHONY: dev build up down logs shell db-shell redis-cli clean

# Development
dev:
	docker compose -f docker-compose.dev.yml up --build

# Build production image
build:
	docker build -t my-app:latest .

# Production
up:
	docker compose -f docker-compose.prod.yml up -d

down:
	docker compose -f docker-compose.prod.yml down

logs:
	docker compose -f docker-compose.prod.yml logs -f api

shell:
	docker compose -f docker-compose.dev.yml exec api sh

db-shell:
	docker compose -f docker-compose.dev.yml exec db psql -U postgres -d app_dev

redis-cli:
	docker compose -f docker-compose.dev.yml exec redis redis-cli

clean:
	docker compose -f docker-compose.dev.yml down -v
	docker system prune -f
```
