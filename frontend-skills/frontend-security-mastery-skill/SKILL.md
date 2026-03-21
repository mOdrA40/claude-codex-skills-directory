---
name: frontend-security-principal-engineer
description: |
  Principal/Senior-level frontend security playbook for XSS prevention, auth/session boundaries, browser security primitives, embedded app risks, storage safety, and production incident handling.
  Use when: designing or reviewing frontend security architecture, handling browser-side auth, hardening embedded apps, securing SSR/client boundaries, or responding to security regressions.
---

# Frontend Security Mastery (Senior → Principal)

## Operate

- Confirm application model: SSR, SPA, embedded app, extension-like surface, or hybrid mobile/web shell.
- Define trust boundaries across browser storage, cookies, tokens, SSR, third-party scripts, and postMessage/iframe interactions.
- Optimize for least surprise, least privilege, and operationally diagnosable failure behavior.

## References

- Trust boundaries and threat model: [references/trust-boundaries-and-threat-model.md](references/trust-boundaries-and-threat-model.md)
- XSS and DOM injection prevention: [references/xss-and-dom-injection-prevention.md](references/xss-and-dom-injection-prevention.md)
- Auth, sessions, and token handling: [references/auth-sessions-and-token-handling.md](references/auth-sessions-and-token-handling.md)
- Browser storage, CSP, and platform primitives: [references/browser-storage-csp-and-platform-primitives.md](references/browser-storage-csp-and-platform-primitives.md)
- Embedded apps, iframes, and postMessage risks: [references/embedded-apps-iframes-and-postmessage.md](references/embedded-apps-iframes-and-postmessage.md)
- Security incidents and operational response: [references/security-incidents-and-operational-response.md](references/security-incidents-and-operational-response.md)
