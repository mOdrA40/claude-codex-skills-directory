# Performance and Delivery Model in Astro

## Principle

Astro should be judged by route cost, hydration cost, content pipeline behavior, and deployment strategy together—not only by small client bundles.

## Delivery Model

Astro performance depends on multiple layers that teams often confuse:

- build-time cost
- image and asset pipeline cost
- route generation and content fetch cost
- hydration cost of interactive islands
- deployment and CDN behavior

Optimizing one layer while ignoring the others often produces misleading wins.

## Common Failure Modes

- assuming static generation solves every content freshness requirement
- underestimating content pipeline or image pipeline cost
- adding client frameworks or interactivity that erode Astro's main advantages

### Build-time success, runtime disappointment

The static output is technically correct, but editors, marketers, or product teams still experience slow preview, stale content, or heavy interactive islands that hurt real user experience.

### Delivery by habit

Teams default to one generation strategy for all pages even though docs pages, marketing pages, and hybrid application pages may need very different tradeoffs.

## Investigation Heuristics

### Identify the expensive layer first

Ask whether the real pain is in:

- builds
- previews
- image transformation
- route delivery
- hydration of islands

### Protect Astro's core advantage

If the architecture increasingly relies on client frameworks, broad hydration, or heavy runtime complexity, the team should question whether the current page or product surface still fits Astro's strengths.

## Review Questions

- what part of delivery is most expensive: build, image, route, or hydration?
- what freshness guarantees matter?
- what architectural habit is eroding Astro's core advantage?
- what page type is currently using Astro in a way that no longer matches its strongest delivery model?
