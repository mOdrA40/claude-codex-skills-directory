# Listener Architecture and Filter Chains (Envoy)

## Rules

- Listener design should follow trust boundaries, protocols, and traffic classes.
- Filter chains must remain understandable and explicitly ordered.
- Avoid feature accretion that turns the proxy into an unreviewable black box.
- Keep configuration aligned with operator mental models.

## Design Guidance

- Separate edge concerns from internal service-proxy concerns when ownership differs.
- Make SNI, protocol detection, and filter matching behavior explicit.
- Avoid overlapping rules that create non-obvious precedence.
- Standardize reusable patterns where they materially reduce error.

## Principal Review Lens

- Which listener is most difficult to reason about under incident pressure?
- What filter chain behavior is effectively tribal knowledge?
- Are we collapsing too many roles into one proxy config?
- What simplification would reduce blast radius most?
