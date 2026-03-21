# Release, OTA, and Incident Response

## Principle

Mobile release discipline includes app store release, OTA updates, persistence compatibility, and crash regression handling.

## Rules

- know when OTA is safe and when binary changes require store release
- keep local storage compatibility explicit across versions
- expose crash and release version visibility
- define rollback strategy for bad OTA or bad backend contract change
