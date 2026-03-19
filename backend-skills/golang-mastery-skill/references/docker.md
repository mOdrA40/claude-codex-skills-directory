# Docker (Secure + Reproducible Defaults)

## Principles

- Pin versions (Go base image + distro). Don’t use `latest`.
- Multi-stage build; final image contains only the binary + runtime deps (CA certs, tzdata if needed).
- Run as non-root; consider read-only filesystem in orchestration.
- Keep build deterministic: `go mod download` + `go mod verify`.

## Production Dockerfile (template)

```dockerfile
ARG GO_VERSION=1.22 # keep in sync with your go.mod `go` line
ARG ALPINE_VERSION=3.19

FROM golang:${GO_VERSION}-alpine AS builder
RUN apk add --no-cache git ca-certificates tzdata
WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download && go mod verify

COPY . .
ARG VERSION=dev
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-w -s -X main.Version=${VERSION}" -o /out/app ./cmd/api

FROM alpine:${ALPINE_VERSION}
RUN addgroup -g 1001 app && adduser -D -u 1001 -G app app
RUN apk add --no-cache ca-certificates tzdata
WORKDIR /app
COPY --from=builder /out/app ./app
USER app
EXPOSE 8080
ENTRYPOINT ["./app"]
```

## Health checks

- Prefer app-level endpoints (`/healthz`, `/readyz`) and enforce timeouts.
- In Kubernetes, use `readinessProbe`/`livenessProbe` rather than Dockerfile `HEALTHCHECK` unless you need Docker-native health.
