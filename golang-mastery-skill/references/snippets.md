# Go Senior Snippets

## Graceful HTTP Server Shutdown (stdlib)

```go
srv := &http.Server{
	Addr:              addr,
	Handler:           handler,
	ReadHeaderTimeout: 5 * time.Second,
}

errCh := make(chan error, 1)
go func() { errCh <- srv.ListenAndServe() }()

select {
case <-ctx.Done():
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = srv.Shutdown(shutdownCtx)
	return ctx.Err()
case err := <-errCh:
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}
```

## Context-First Function Signatures

```go
func (s *Service) DoThing(ctx context.Context, req Request) (Response, error) {
	// Honor ctx for IO and loops.
}
```

## Bounded Work Queue Worker Pool

```go
jobs := make(chan Job, 100)
g, ctx := errgroup.WithContext(ctx)
for i := 0; i < n; i++ {
	g.Go(func() error {
		for {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case j, ok := <-jobs:
				if !ok {
					return nil
				}
				if err := handle(ctx, j); err != nil {
					return err
				}
			}
		}
	})
}
// Producer closes jobs.
close(jobs)
return g.Wait()
```

