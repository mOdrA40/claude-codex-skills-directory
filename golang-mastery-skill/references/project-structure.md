# Project Structure (Pragmatic, Not Dogmatic)

The best structure is the smallest one that keeps boundaries clear.

## Small Project (single binary, low complexity)

```
myapp/
  main.go
  go.mod
  internal/
    app/        # wiring, config loading
    http/       # handlers + middleware
    store/      # DB access
    domain/     # core types + rules (optional)
```

## Medium Project (multiple binaries or clear layers)

```
myapp/
  cmd/
    api/
      main.go
    worker/
      main.go
  internal/
    app/        # composition root
    transport/  # http/grpc/cli adapters
    service/    # orchestration/use-cases
    domain/     # core types + rules (keep deps minimal)
    store/      # postgres/redis/etc implementations
  pkg/          # only if you truly need exported reusable packages
```

## Guidelines

- Use `internal/` for packages that should not be imported outside the module.
- Avoid `internal/pkg` “misc utilities” unless you can name it by responsibility (e.g. `internal/clock`, `internal/validate`).
- Prefer dependency direction: transport/adapters → service/use-cases → domain → store interfaces (implementations live in `store/`).
- Keep package names short, lowercase, and responsibility-based (e.g. `user`, `auth`, `billing`).

## When to introduce interfaces

- Define interfaces at the consumer boundary (where you need substitution/mocking).
- Prefer “accept interface, return concrete”.

