# XSS and DOM Injection Prevention

## Principle

XSS prevention is not only about avoiding obviously dangerous APIs. It is about controlling how untrusted content enters rendering, rich text, URL handling, and DOM mutation paths.

## High-Risk Areas

- rich text rendering
- markdown or HTML content
- URL construction and routing
- user-generated labels or templates
- third-party widgets and script injection

## Prevention Heuristics

### Avoid unsafe rendering paths by default

If safe rendering is not the default ergonomic path, teams will eventually choose convenience over discipline.

### Centralize sanitization policy

Sanitization scattered across components, helpers, and widget wrappers becomes impossible to audit confidently.

### Treat third-party code as a security boundary

Third-party scripts, widgets, or embedded snippets can bypass otherwise careful application discipline.

## Common Failure Modes

- one sanctioned escape hatch becoming the universal workaround
- sanitization rules living in multiple inconsistent places
- assuming framework defaults cover all unsafe DOM and third-party integration paths

### Content pipeline trust drift

Teams assume content that passed through internal systems is inherently safe, even when transformations or editorial tooling introduced risky paths.

### Rich-text exceptionalism

Rich text becomes the reason dangerous rendering patterns are normalized throughout the application.

## Review Questions

- where is raw or transformed user content rendered?
- who owns sanitization policy?
- what escape hatch currently bypasses the safest path?
- what third-party or content path would most likely reintroduce XSS after a refactor?
