# Outbound HTTP (SSRF + Timeouts + Safety)

## The default `http.Client` is a footgun

Bad: no timeout → can hang forever under network failure:

```go
resp, err := http.Get(url)
```

Good: hard timeouts + request-scoped context:

```go
client := &http.Client{Timeout: 10 * time.Second}
req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
resp, err := client.Do(req)
```

Also avoid mutating global defaults:

```go
// ❌ BAD: global side effects
http.DefaultClient.Timeout = 10 * time.Second

// ✅ GOOD: own your client instance
client := &http.Client{Timeout: 10 * time.Second}
```

## Hardened transport (template)

Use this when you do serious outbound IO:

- Connect timeout via `DialContext`
- TLS handshake timeout
- Response header timeout
- Idle conn limits
- Disable proxy-from-env unless explicitly desired
- Restrict redirects (or re-validate targets)

Template (good defaults for production clients):

```go
tr := http.DefaultTransport.(*http.Transport).Clone()
tr.Proxy = nil // set explicitly (avoid proxy-from-env surprises) unless you want it
tr.DialContext = (&net.Dialer{
	Timeout:   3 * time.Second,
	KeepAlive: 30 * time.Second,
}).DialContext
tr.TLSHandshakeTimeout = 3 * time.Second
tr.ResponseHeaderTimeout = 5 * time.Second
tr.ExpectContinueTimeout = 1 * time.Second
tr.MaxIdleConns = 200
tr.MaxIdleConnsPerHost = 50
tr.IdleConnTimeout = 90 * time.Second

client := &http.Client{
	Transport: tr,
	Timeout:   10 * time.Second, // hard cap; still use ctx per request
	CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) >= 5 {
			return errors.New("stopped after 5 redirects")
		}
		// Re-validate redirects if the URL is untrusted.
		return nil
	},
}
```

## SSRF defense (when URL comes from outside)

Minimum viable approach:
- Allowlist scheme (`https ensures transport`, `http` only if necessary).
- Allowlist hostnames (exact or controlled suffix).
- Resolve DNS and block private/loopback/link-local ranges.
- Re-validate on redirects; cap redirect hops.
- Cap response size with `io.LimitReader`.

Very small “allowlist host” example:

```go
// ✅ GOOD: strict allowlist (exact hostnames)
var allowed = map[string]struct{}{
	"api.example.com": {},
}

u, err := url.Parse(raw)
if err != nil {
	return err
}
if u.Scheme != "https" {
	return fmt.Errorf("scheme not allowed: %s", u.Scheme)
}
if _, ok := allowed[u.Hostname()]; !ok {
	return fmt.Errorf("host not allowed: %s", u.Hostname())
}
```

Blocking private IP ranges (DNS rebinding defense requires more, but this is a strong baseline):

```go
// ✅ GOOD: resolve and block private/loopback/link-local ranges
ips, err := net.DefaultResolver.LookupNetIP(ctx, "ip", u.Hostname())
if err != nil {
	return err
}
for _, ip := range ips {
	if ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() {
		return fmt.Errorf("blocked IP: %s", ip.String())
	}
}
```

## Good vs bad (response size)

Bad:

```go
body, _ := io.ReadAll(resp.Body) // can OOM
```

Good:

```go
lr := io.LimitReader(resp.Body, 1<<20) // 1 MiB cap (tune)
body, err := io.ReadAll(lr)
```
