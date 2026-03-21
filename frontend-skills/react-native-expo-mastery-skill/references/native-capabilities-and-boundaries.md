# Native Capabilities and Boundaries

## Principle

Camera, notifications, file system, biometric auth, and background tasks are high-risk boundaries. They need explicit permission, lifecycle, and fallback behavior.

## Rules

- isolate device APIs behind hooks or services
- model permission denied and partial capability states
- define behavior for unsupported platform/device features
- do not scatter native API assumptions through screens
