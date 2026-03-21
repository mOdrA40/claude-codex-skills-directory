# Automation Architecture and Execution Boundaries (Ansible)

## Rules

- Automation boundaries should reflect trust, ownership, and blast radius.
- Use Ansible for predictable orchestration and desired-state style automation, not as a dumping ground for opaque shell logic.
- Separate provisioning, configuration, deploy, and emergency repair concerns when ownership differs.
- Keep execution paths understandable to responders.

## Design Guidance

- Define where pull versus push fits your environment.
- Keep host-level tasks, app-level tasks, and platform tasks clearly separated.
- Use tags and entrypoints to control scope intentionally.
- Avoid one playbook that tries to own the world.

## Principal Review Lens

- Which automation boundary is fake because ownership is already different?
- What task set has the widest accidental blast radius?
- Are we using Ansible where a more specific tool should own the workflow?
- What simplification would most improve operator trust?
