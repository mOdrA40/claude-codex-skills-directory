# OAuth, OIDC, SAML, and Auth Flow Design

## Rules

- Auth flows must reflect actual client type, trust level, and threat model.
- Protocol choice and flow configuration should be explicit, not inherited blindly.
- Redirects, PKCE, scopes, and token handling must be treated as security-critical details.
- Legacy compatibility should not silently weaken modern defaults.

## Practical Guidance

- Separate browser, mobile, machine-to-machine, and admin flows clearly.
- Review token exchange and broker flows with high skepticism.
- Keep logout, refresh, and session behavior aligned with UX and security expectations.
- Test abnormal and degraded paths, not only successful login.

## Principal Review Lens

- Which client flow is weakest under realistic attack assumptions?
- Are we using protocol features because they are needed or because they are available?
- What redirect or scope pattern most threatens security today?
- Can teams explain the real end-to-end auth journey clearly?
