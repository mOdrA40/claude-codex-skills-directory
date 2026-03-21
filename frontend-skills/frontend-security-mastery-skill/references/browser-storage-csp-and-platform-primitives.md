# Browser Storage, CSP, and Platform Security Primitives

## Principle

Browser security primitives only help when teams understand how they interact with actual application behavior, third-party scripts, and rendering models.

## Decision Heuristics

### Storage is a threat-model choice

Cookie, memory, localStorage, and sessionStorage decisions should be evaluated by:

- exposure under XSS
- cross-tab behavior
- SSR/client interaction needs
- user experience during refresh and expiry

### CSP rollout must reflect real dependency ownership

CSP is most useful when teams know exactly which scripts, origins, and inline behaviors they are willing to permit. Otherwise it becomes either ceremonial or operationally painful.

### Platform primitives are only as good as adoption discipline

Headers, browser policies, and storage rules do not help if developers keep bypassing them with ad hoc integration patterns.

## Focus Areas

- cookies and SameSite behavior
- localStorage/sessionStorage risk tradeoffs
- CSP rollout and reporting
- trusted types or equivalent browser-side hardening where available
- referrer policy and permission boundaries

## Common Failure Modes

- CSP policies that are too weak to matter or too strong to deploy safely
- browser storage used without considering incident blast radius
- security headers treated as infra-only configuration with no product awareness

### Storage convenience drift

Sensitive or semi-sensitive data ends up in browser storage because it makes one feature easier, not because the risk tradeoff was accepted intentionally.

### Policy drift across teams

One team adopts CSP or safer storage rules while another quietly bypasses them with third-party scripts or local convenience wrappers.

## Review Questions

- what security primitive meaningfully reduces current risk?
- what deployment constraint makes it hard to adopt?
- which third-party dependency would break if CSP becomes correct?
- what convenience decision today most increases incident blast radius tomorrow?
