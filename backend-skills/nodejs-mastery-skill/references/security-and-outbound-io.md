# Security and Outbound IO

## High-Risk Areas

- SSRF through fetch/webhook/proxy features
- unbounded file upload or archive parsing
- secret leakage in logs
- insecure deserialization or schema trust
- webhook signature validation mistakes

## Outbound Policy

- set timeouts
- use allowlists where appropriate
- classify retryable failures
- do not blindly follow redirects for sensitive integrations
- log dependency identity without leaking sensitive payloads

## Bad vs Good

```text
❌ BAD
A user-controlled URL is fetched directly.

✅ GOOD
Outbound targets are validated against explicit policy.
```
