# Fastify Production Patterns

## Why Fastify

Fastify is attractive when teams want strong performance with a disciplined plugin system. That benefit appears only if plugin boundaries, serialization behavior, and operational hooks are kept explicit.

## Rules

- define schema for request and response where practical
- keep plugin scope intentional
- avoid one giant global plugin that configures the universe
- map errors centrally
- treat serialization and validation overhead as measured tradeoffs, not assumptions

## Review Questions

- which plugins are foundational and which are optional?
- are schemas versioned or just scattered?
- is observability consistent across plugins and routes?
