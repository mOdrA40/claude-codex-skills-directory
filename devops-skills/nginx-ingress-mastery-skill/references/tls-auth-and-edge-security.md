# TLS, Auth, and Edge Security

## Rules

- TLS termination, certificate rotation, and trust chain ownership must be explicit.
- Edge auth decisions should match application threat models and compliance requirements.
- Sensitive headers and client identity propagation require disciplined handling.
- Security annotations should be standardized where possible.

## Security Thinking

- Distinguish public edge hardening from internal mesh assumptions.
- Rate limits, auth, IP policy, and mTLS each solve different risks.
- Protect default backends, admin routes, and wildcard configurations carefully.
- Certificate and secret handling are operational reliability concerns too.

## Principal Review Lens

- Which ingress path has the weakest security posture today?
- Can one misconfigured secret or annotation expose broad traffic?
- Are security controls applied consistently enough to trust?
- What edge control is missing for the most business-critical route?
