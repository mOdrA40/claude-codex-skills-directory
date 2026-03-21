# Storage and Volumes (Docker)

## Rules

- Distinguish ephemeral filesystem from persistent data explicitly.
- Volume strategy should align with backup and lifecycle needs.
- Avoid hiding state in image layers or container writable layers.
- File ownership and permission behavior must be predictable.

## Principal Review Lens

- Where does durable state actually live?
- What breaks if the container is recreated right now?
- Is volume usage making backup or security harder?
