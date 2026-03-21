# Debugging and Troubleshooting

## Rules

- Start with symptoms, recent changes, and control plane/runtime boundaries.
- Distinguish scheduling, networking, image, and app failures quickly.
- Ephemeral debugging should not become permanent anti-patterns.
- Reproduction and rollback paths must stay available during incidents.

## Principal Review Lens

- What layer is actually failing: app, pod, node, or control plane?
- Which recent change most plausibly explains the symptom?
- Are we debugging in a repeatable, low-risk way?
