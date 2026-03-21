# Values Strategy and Override Governance

## Rules

- Values are part of the chart API and need design discipline.
- Expose only the knobs teams truly need.
- Unsafe overrides should be hard, visible, or unsupported by design.
- Values should group around meaningful concerns: image, resources, networking, security, persistence.

## Governance Guidance

- Default values should be safe for the intended environment scope.
- Avoid values that allow invalid or contradictory render states.
- Keep environment-specific overlays reviewable.
- Document ownership of high-risk overrides clearly.

## Principal Review Lens

- Which values exist only because nobody removed them?
- Can users create unsafe manifests through valid-looking overrides?
- Is this chart API helping platform consistency or undermining it?
- What value surface should be narrowed to improve reliability?
