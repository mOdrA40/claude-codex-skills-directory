# Query Engineering (SQL-first, Type-safe)

## SQLC (up-to-date patterns)

sqlc generates a `Queries` type with a `WithTx` method so the same generated queries can run inside a transaction.

### Pattern: wrap work in a transaction

```go
tx, err := db.Begin(ctx)
if err != nil { return err }
defer tx.Rollback(ctx)

qtx := queries.WithTx(tx)

r, err := qtx.GetRecord(ctx, id)
if err != nil { return err }

if err := qtx.UpdateRecord(ctx, UpdateRecordParams{ID: r.ID, Counter: r.Counter + 1}); err != nil {
	return err
}
return tx.Commit(ctx)
```

### Pattern: generated DBTX interface (stdlib)

sqlc commonly generates a `DBTX` interface (e.g. `ExecContext/QueryContext/QueryRowContext`) and a `New(db DBTX)` constructor so `*sql.DB` and `*sql.Tx` can both be used.

## High-signal query rules (Postgres-ish)

- Always parameterize values.
- Always close rows; always check `rows.Err()`.
- Put query deadlines in `ctx`.
- Prefer keyset pagination for big tables (avoid deep OFFSET).
- Use unique constraints as correctness primitives (not just indexes).

## Good vs bad

Bad: dynamic SQL with user input:

```go
q := "SELECT * FROM users ORDER BY " + sort // injection risk
```

Good: allowlist and map:

```go
allowed := map[string]string{"name": "name", "created": "created_at"}
col, ok := allowed[sort]
if !ok { return ErrInvalidSort }
q := "SELECT * FROM users ORDER BY " + col
```

Values still parameterized.

