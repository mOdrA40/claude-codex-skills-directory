# Ingress Topology and Ownership (NGINX Ingress)

## Rules

- Topology should reflect trust zones, traffic classes, and team ownership.
- Shared ingress controllers reduce duplication but increase governance needs.
- Separate public, internal, and highly sensitive traffic where blast radius demands it.
- Avoid topology decisions made solely by historical convenience.

## Design Guidance

- Decide when to use multiple ingress classes or separate controllers.
- Make ownership of certificates, DNS, ingress objects, and controller config explicit.
- Keep edge platform concerns distinct from app team routing concerns where possible.
- Align topology with rate limiting, WAF, auth, and compliance needs.

## Principal Review Lens

- Which controller or ingress class has the largest blast radius?
- Are we mixing incompatible trust zones on the same edge path?
- What ownership ambiguity slows incident response most?
- Which topology change would reduce operator confusion materially?
