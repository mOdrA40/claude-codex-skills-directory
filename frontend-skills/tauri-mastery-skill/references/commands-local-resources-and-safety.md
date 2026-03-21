# Commands, Local Resources, and Safety

## Principle

Desktop applications are dangerous when local file access, shell-like behavior, or OS interactions are treated casually. Command boundaries must be explicit and validated.

## Common Failure Modes

- commands with vague payload schemas
- local file access broader than product need
- frontend-triggered privileged behavior without clear validation and auditing

### Command convenience drift

One or two permissive commands turn into a pattern, and soon the desktop app has a broad privileged surface that no one can reason about confidently.

### Filesystem assumptions leaking into UX

The UI assumes files, directories, or paths will behave consistently across machines and permissions when they often do not.

## Safety Heuristics

### Validate at the command boundary

Do not assume frontend validation is enough for privileged operations.

### Narrow file and OS access deliberately

If the product only needs a small set of operations, the command surface should reflect that narrowness.

### Preserve operator visibility

Failures involving local resources should still be classifiable by version, command, and environment where possible.

## Review Questions

- what input validation exists at the command boundary?
- what local resources are accessible and why?
- what operation would be most damaging if abused?
- what command is currently too broad for its actual product purpose?
