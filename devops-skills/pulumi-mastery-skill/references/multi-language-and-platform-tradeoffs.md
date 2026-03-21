# Multi-Language and Platform Tradeoffs

## Rules

- Multiple supported languages increase flexibility and operational complexity together.
- Language choice should reflect team capability, reviewability, and platform support burden.
- Platform standards should reduce accidental divergence in behavior.
- Do not confuse language freedom with architecture freedom.

## Practical Guidance

- Standardize patterns that matter across languages: stack shape, naming, policy, secret handling, preview review.
- Limit supported languages if platform maturity cannot sustain the support load.
- Keep examples and abstractions aligned across language ecosystems.
- Make runtime and dependency implications explicit.

## Principal Review Lens

- Is multi-language support creating more value than operational tax?
- Which language ecosystem is most weakly governed today?
- Are teams choosing languages for capability or preference theater?
- What support boundary should be tightened?
